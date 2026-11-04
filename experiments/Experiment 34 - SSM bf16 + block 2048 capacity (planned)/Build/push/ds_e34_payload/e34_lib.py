"""E34 library (2026-11-03): the GDN SSM state in bf16 instead of fp32, with KV block 2048 instead of 4096, on the DFlash2
production engine (NB-D1's base B = E29_DFLASH: NBDF-K100 D3s, DFlash2 K3, SW-1024 BF16 drafter KV, FLASH_ATTN, seam
341d859b + GRD, Q-defer, seqs 24, pin 7,933,526,016, DFLASH_DRAFT_EMBED_HOST=1 from the first try, QSA 128 / 64).

Arms (this order = the run order; each = B + its change; no cross-arm skip):
  B       none (fp32 SSM, block 4096, SW drafter block 1024, 24 seqs)
  S16     + --mamba-ssm-cache-dtype bfloat16 (block 4096: the Mamba page is padded to the 2 MiB attention page, so the pool
          and F are unchanged (CPU planner 540 / F 12 = B); accuracy + speed with no layout change)
  S16B2   + --mamba-ssm-cache-dtype bfloat16 --block-size 2048 + DFLASH_DRAFT_SW_BLOCK=512 (the drafter SW block must follow:
          at SW 1024 its 10,485,760 B page sets the PP1 stride and the pool falls to 756 blocks of 2048 (CPU planner); at 512
          the page is 5,242,880 B and the pool is 1,080 blocks of 2048 = the same tokens as B, F 17 instead of 12 blocks
          but 2048-token blocks: 34,816 instead of 49,152 token-equivalents a request)
  S16B2W  S16B2 + --max-num-seqs W (W from S16B2's measured pool: the static admission bound at 40k, clamp 24 .. 40;
          captures every multiple of 4 up to 4 W); planned with the CPU planner's W 29, re-planned at run time
Every arm: boot read-backs (the server's cache_config.mamba_ssm_cache_dtype + block size per worker, the GDN spec state
dtype, the drafter SW block, 'kv cache group sizes', 'Padding mamba page size', 'GPU KV cache size', 'Maximum
concurrency', pool, F + its token-equivalent), smoke (no ladder), NBDF's greedy oracle vs B (B: its own A/A), NBDF's long
oracle 90k + 116k vs B, NBDF's prefix cold / warm multi-turn test, the windows (90k and ~116k always), NBDF's teacher-forced
prompt logprobs vs B (B: A/A) on 8k / 40k task prompts + three recorded Flash-Next thinking traces of 69k / 94k / 113k
tokens (E31 direct), a [LIVE] line after every window and test. Real failures only (crash, OOM, a flag rejected or not
engaged, NaN, zero accept); every KL / oracle / speed label is recorded only.

The never-do entry `--mamba-ssm-cache-dtype bfloat16` (nb4t_lib.NEVER_DO_ARG_VALUES, "C3 SSM bf16: parked (0 capacity + B2
FAIL)", E25 NB1 R2 2026-10-31: zero capacity at block 4096 and one FAIL_HIGH_MARGIN oracle row in a combined B2 boot) is
lifted for exactly the three SSM arms of this notebook (the user's E34 request): launch_check / check_arm_plan drop exactly
that one violation string on an arm whose plan has the flag; every other never-do item still refuses.

Pure logic, stdlib only; nbd1_lib (D1), nbdf_lib (N), nb4t_lib (T) are handed in as module objects (never imported or edited).
"""
import copy
import json
import math
import re

LIB_VERSION = "e34-lib 2026-11-03 r1 (B / S16 / S16B2 / S16B2W on the DFlash2 production base; SSM bf16 + block 2048)"
BASE_KEY = "B"
ARM_KEYS = ("B", "S16", "S16B2", "S16B2W")
LEVERS = ARM_KEYS[1:]
SSM_ARMS = ("S16", "S16B2", "S16B2W")
B2_ARMS = ("S16B2", "S16B2W")
W_ARMS = ("S16B2W",)
ITEM_KEYS = ("BMAX", "SMOKE")                       # BMAX: B's own max-resident windows; SMOKE: S16B2's smoke plan
SWITCH_KEYS = ARM_KEYS + ITEM_KEYS
SWITCH_DEFAULT = {"B": True, "S16": True, "S16B2": True, "S16B2W": True, "BMAX": True, "SMOKE": False}   # the kit notebook
COPIES = {"full": dict(SWITCH_DEFAULT),
          "smoke": {"B": False, "S16": False, "S16B2": True, "S16B2W": False, "BMAX": False, "SMOKE": True}}
OWNERS = {"full": "foysalemonshanto", "smoke": "foysalemonshanto"}
PORTS = {"B": 8011, "S16": 8231, "S16B2": 8251, "S16B2W": 8271}
SSM_FLAG, BLOCK_FLAG, SEQS_FLAG = "--mamba-ssm-cache-dtype", "--block-size", "--max-num-seqs"
SSM_ARGS = [SSM_FLAG, "bfloat16"]
B2_ARGS = [BLOCK_FLAG, "2048"]
ENV_SW = "DFLASH_DRAFT_SW_BLOCK"
SW_B, SW_B2 = "1024", "512"
LIFTED = "argv --mamba-ssm-cache-dtype bfloat16"    # nb4t_lib.never_do_violations' exact string for that entry
MERLIN_BLOCK = 4096                                 # tools/merlin/launch.py:334 (BLOCK_SIZE knob default) + :474 --block-size
PLAN = {"B": {"ssm": "float32", "block": 4096, "sw": 1024}, "S16": {"ssm": "bfloat16", "block": 4096, "sw": 1024},
        "S16B2": {"ssm": "bfloat16", "block": 2048, "sw": 512}, "S16B2W": {"ssm": "bfloat16", "block": 2048, "sw": 512}}
PIN = 7933526016
SEQS = 24
W_BAND_TOKENS, W_MIN, W_MAX = 40000, 24, 40
# Merlin's real KV planner on CPU (e34_capacity_probe.py, the klee100 K3 GPU worker record, pin 7,933,526,016): recorded
# predictions, the kit's KV probe expectations and the W planned at build time
CPU_PLANNER = {"B": {"pool": 540, "stride": [14680064, 14680064, 14155776, 14155776], "F_adm": 12, "F_steady": 10, "mc_131072": 12.25},
               "S16": {"pool": 540, "stride": [14680064, 14680064, 14155776, 14155776], "F_adm": 12, "F_steady": 10, "mc_131072": 12.25},
               "S16B2": {"pool": 1080, "stride": [7340032, 7340032, 7077888, 7077888], "F_adm": 17, "F_steady": 12, "mc_131072": 13.32},
               "S16B2_SW1024": {"pool": 756, "stride": [7340032, 7340032, 10485760, 10485760], "F_adm": 12, "F_steady": 10, "mc_131072": 9.93},
               "fp32_block2048": "BLOCK_RAISED (the 1,732,992 B fp32 Mamba state page > the 1,048,576 B attention page at 2048)"}
W_PRED = 29                                         # (1080 - 2) // (ceil(40000 / 2048) + 17)
GDN_STATE_BYTES = {"float32": 1732992, "bfloat16": 946560}   # conv bf16 [6, 5120] + SSM [24, 128, 128] + replay fp32 [24, 4, 257]
ATTN_PAGE_PER_TOKEN = 512                           # FullAttentionSpec 1 head x 256 x fp8 (K + V) per layer: the Mamba pad target
TITLES = {"B": "B = E29_DFLASH production (NB-D1's B): NBDF-K100 D3s + DFLASH_DRAFT_EMBED_HOST=1 + QSA 128 / 64 (fp32 SSM, block 4096, SW 1024, 24 seqs)",
          "S16": "B + --mamba-ssm-cache-dtype bfloat16 (block 4096)",
          "S16B2": "B + --mamba-ssm-cache-dtype bfloat16 --block-size 2048 + DFLASH_DRAFT_SW_BLOCK=512",
          "S16B2W": "S16B2 + --max-num-seqs W (S16B2's measured pool, static bound at 40k, 24 .. 40)"}
KIT_KEYS = ("e34_arm_plan_ok", "e34_launch_path_all_arms_ok", "e34_help_flag_gate_real_ok", "e34_argparse_change_per_arm",
            "e34_kv_probes_ok", "e34_launch_env_ok", "e34_captures_argv_packviews_ok", "e34_switch_default_ok",
            "e34_neverdo_lift_exact", "e34_merlin_source_ok", "e34_tf_long_ok")

# ------------------------------------------------------------------ windows / items
B_WINDOWS = [["8k", 16, "#1"], ["8k", 16, "#2"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]]
LEVER_WINDOWS = [["8k", 16, "#1"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]]


def max_windows(n, bands=("20k", "40k", "90k", "116k")):
    """The max-resident windows (tag W): level n, capped at run time by the boot's resident bound (D1.CAPPED_BANDS)."""
    return [[b, int(n), "W"] for b in bands]


FULL_B_ITEMS = {"ladder": False, "prefill_probe": True, "oracle": True, "long_oracle": True, "prefix": True, "sibling": False, "tf": True}
FULL_LEVER_ITEMS = dict(FULL_B_ITEMS, prefill_probe=False)
SMOKE_ITEMS = {"ladder": False, "prefill_probe": False, "oracle": False, "long_oracle": True, "prefix": False, "sibling": False, "tf": True}
TF_SYNTH = ("8k", "40k")                            # RC2's TF variant 5 (NBDF-K100 tf_bands minus 2k)
TF_LONG = (("tf69k", "5545f144"), ("tf94k", "409aa875"), ("tf113k", "0934a4d8"))   # E31 direct recorded traces (prompt + p1)
TF_SMOKE = ("tf113k",)
TF_LONG_FILE = "e34_tf_long.json"
TF_LONG_DS = "foysalemonshanto/q27-e34-tf-long"


def switch_map(on, keys=SWITCH_KEYS):
    """E34_ON -> {key: bool}: absent / not a dict -> SWITCH_DEFAULT; a dict -> bool(on[k]), a missing key -> off."""
    if on is None or not isinstance(on, dict):
        return dict(SWITCH_DEFAULT)
    return {k: bool(on.get(k, False)) for k in keys}


def switch_record(on):
    m = switch_map(on)
    return {"E34_ON": on if isinstance(on, (dict, type(None))) else repr(on)[:200], "effective": m, "absent": on is None,
            "invalid_not_a_dict": on is not None and not isinstance(on, dict),
            "unknown_keys": sorted(str(k) for k in set(on) - set(SWITCH_KEYS)) if isinstance(on, dict) else [],
            "off": [k for k, v in m.items() if not v]}


# ------------------------------------------------------------------ the time model (minutes)
# GPU data: E25 NB-S1 full (this boot path: first ready 714 s, warm 252-282 s, windows 8k@16 125 s .. 116k@6 272 s) and the
# NBDF-K100 Q GPU run 2026-11-01 (oracle 0.28 + A/A 0.30, long oracle 0.52, prefix test 1.94, warmup 0.76 min). TF: not
# GPU-measured on this path; one pass = 324k prompt tokens at ~2,500 tok/s + the prompt-logprob JSON (estimate 2.7 min,
# the 113k trace alone 1.0). An argv change of the cache config recompiles (block / dtype / captures in the compile key).
SETUP_MIN = 6.0
READY_MIN = {"first": 12.0, "recompile": 7.0, "recompile_block": 7.5, "recompile_captures": 8.5}
READY_CLASS = {"B": "first", "S16": "recompile", "S16B2": "recompile_block", "S16B2W": "recompile_captures"}
ITEM_MIN = {"facts": 1.5, "smoke": 1.3, "prefill_probe": 0.5, "oracle": 0.4, "oracle_aa": 0.4, "long_oracle": 0.7, "prefix": 2.0,
            "warmup": 1.0, "tf_pass": 2.7, "tf_pass_smoke": 1.0, "end": 1.2}
WINDOW_MIN = {"8k@16": 2.1, "20k@24": 3.65, "40k@16": 5.85, "90k@8": 4.6, "116k@6": 4.55}
POST_MIN = 0.8
BAND_TOK = {"8k": 8192, "20k": 20000, "40k": 40000, "90k": 90000, "116k": 116000}
WIN_SEC, ADMIT_TOK_S, LONG_RESET_SEC = 90.0, 3000.0, 60.0
START_RESERVE_MIN = 5.0
# the predicted resident levels of the W windows (nbdf_lib.level_for_band on the CPU planner's pool / mc at prefill 3000 tok/s,
# WIN 90 s; the builder asserts them): B 540 / 12.25 at block 4096, S16B2 1080 / 13.32 at block 2048 (seqs W_PRED)
PRED_LEVELS = {"B": {"20k": 24, "40k": 20, "90k": 13, "116k": 11}, "S16B2W": {"20k": 29, "40k": 22, "90k": 14, "116k": 12}}


def _c(x):
    return int(math.ceil(round(x, 6)))


def win_need_sec(band, n):
    """= the body's win_need() at the admission rate min(prefill, 3000 tok/s)."""
    return WIN_SEC + 120 + n * BAND_TOK[band] / ADMIT_TOK_S


def window_min(band, n):
    """A measured window when one exists (NB-S1 full), else admission + window + 40 s (+ 1 min long-band reset)."""
    k = f"{band}@{int(n)}"
    if k in WINDOW_MIN:
        return WINDOW_MIN[k]
    return round((n * BAND_TOK[band] / ADMIT_TOK_S + WIN_SEC + 40.0) / 60.0 + (1.0 if band in ("90k", "116k") else 0.0), 2)


def body_items(key, windows, items, levels=None, smoke=False):
    """[(name, min, need_sec | None)] in body order (arm_body: facts, smoke, prefill, oracle (+ A/A on B), long oracle,
    prefix, warmup, windows, TF (B: ref + A/A), end)."""
    it = [("facts", ITEM_MIN["facts"], None), ("smoke", ITEM_MIN["smoke"], 240)]
    if items.get("prefill_probe"):
        it.append(("prefill_probe", ITEM_MIN["prefill_probe"], 300))
    if items.get("oracle"):
        it.append(("oracle", ITEM_MIN["oracle"], 240))
        if key == BASE_KEY:
            it.append(("oracle_aa", ITEM_MIN["oracle_aa"], None))
    if items.get("long_oracle"):
        it.append(("long_oracle", ITEM_MIN["long_oracle"], 300))
    if items.get("prefix"):
        it.append(("prefix", ITEM_MIN["prefix"], 240))
    if windows:
        it.append(("warmup", ITEM_MIN["warmup"], 150))
    for b, n, tag in windows:
        lv = int(n)
        if tag == "W" and levels:
            lv = min(lv, int(levels.get(b, lv)))
        nd = win_need_sec(b, lv) + (LONG_RESET_SEC if b in ("90k", "116k") else 0.0)
        it.append((f"{b}@{lv}{tag}", window_min(b, lv), nd))
    if items.get("tf"):
        passes = 2 if (key == BASE_KEY or smoke) else 1
        per = ITEM_MIN["tf_pass_smoke"] if smoke else ITEM_MIN["tf_pass"]
        it.append(("tf", passes * per, 240))
    it.append(("end", ITEM_MIN["end"], None))
    return it


def body_plan(items):
    """(body_min, cap_min, rows): the cap holds every gated item's due() need from its planned start, + 1 min."""
    t, need, rows = 0.0, 0.0, []
    for nm, m, nd in items:
        if nd is not None:
            need = max(need, t + nd / 60.0)
        rows.append([nm, m, None if nd is None else round(nd / 60.0, 2), round(t, 2)])
        t += m
    return round(t, 2), _c(max(t, need) + 1.0), rows


def arm_time(key, windows, items, ready_class=None, levels=None, smoke=False):
    body, cap, rows = body_plan(body_items(key, windows, items, levels, smoke))
    rc = ready_class or READY_CLASS[key]
    return _c(READY_MIN[rc] + body + POST_MIN), cap, {"ready": READY_MIN[rc], "ready_class": rc, "body": body, "post": POST_MIN, "items": rows}


def copy_walk(arms, switch, budget_min):
    sw = switch_map(switch)
    left = budget_min - SETUP_MIN
    walk, runtime = [], SETUP_MIN
    for k, a in arms.items():
        if not sw.get(k, True):
            continue
        fits = left >= a["est_min"] + START_RESERVE_MIN
        walk.append({"arm": k, "est_min": a["est_min"], "cap_min": a["cap_min"], "left_min": round(left, 1), "runs": fits})
        if fits:
            left -= a["est_min"]
            runtime += a["est_min"]
    return {"walk": walk, "runtime_min": round(runtime, 1), "left_end_min": round(left, 1), "runs": [w["arm"] for w in walk if w["runs"]],
            "skipped": [w["arm"] for w in walk if not w["runs"]], "all_run": all(w["runs"] for w in walk), "budget_min": budget_min}


def copy_need_min(arms, switch):
    sw = switch_map(switch)
    t, end = SETUP_MIN, SETUP_MIN
    for k, a in arms.items():
        if not sw.get(k, True):
            continue
        tm = a["time_model"]
        end = max(end, t + tm["ready"] + a["cap_min"] + tm["post"])
        t += a["est_min"]
    return round(end, 1)


def budget_for(arms, switch):
    """The copy's hard stop: max(planned runtime, full-cap end) + 5 min, rounded up to 5 (NB-D1's rule)."""
    w = copy_walk(arms, switch, 10 ** 6)
    top = max(w["runtime_min"], copy_need_min(arms, switch)) + START_RESERVE_MIN
    return int(5 * -(-top // 5))


# ------------------------------------------------------------------ the arm plan
def _sw_key(a):
    return str((a.get("env") or {}).get(ENV_SW))


def expect_of(D1, N, a):
    return D1.expect_of(N, a)


def w_arm(D1, N, b, w):
    """S16B2W at max_num_seqs w: B's base argv + its spec pair + the captures every multiple of 4 up to 4 w + --max-num-seqs
    w (NB-D1's seqs-arm shape) + the S16B2 argv; env = S16B2's (SW 512)."""
    w = int(w)
    base_extra, tail = D1._split_tail(b["extra"])
    caps = D1.dflash_captures(w, 3)
    a = copy.deepcopy(b)
    a["extra"] = base_extra + tail[:2] + ["--compilation-config", N.compilation_json(caps), SEQS_FLAG, str(w)] + SSM_ARGS + B2_ARGS
    a["env"][ENV_SW] = SW_B2
    a["captures"], a["streams"] = caps, w
    a["expect_cfg"].update(max_num_seqs=w, captures=caps)
    a["coverage"] = N.coverage(caps, w, 3)
    a["e34"] = dict(PLAN["S16B2W"], seqs=w)
    a["d1_change"] = {"kind": "e34", "env": {ENV_SW: SW_B2}, "argv": SSM_ARGS + B2_ARGS + [SEQS_FLAG, str(w)], "captures": caps}
    a["expect"] = D1.expect_of(N, a)
    return a


def arm_plan(D1, N, d3s, seam_env_k4=None, switch=None, pin=PIN, seqs=SEQS, w=W_PRED):
    """d3s = NBDF-K100's D3s. B = NB-D1's B (D1.arm_plan, unchanged engine) with E34's items / windows / time; each lever =
    a deep copy of B + its change. SMOKE on: S16B2's plan becomes the smoke plan (no windows, no oracle / prefix, the long
    oracle + one 113k TF trace)."""
    sw = switch_map(switch)
    d1 = D1.arm_plan(N, d3s, seam_env_k4=seam_env_k4, switch=None, pin=pin, seqs=seqs)
    b = copy.deepcopy(d1[BASE_KEY])
    bw = copy.deepcopy(B_WINDOWS) + (max_windows(W_PRED, ("40k", "90k", "116k")) if sw.get("BMAX") else [])
    est, cap, tm = arm_time(BASE_KEY, bw, FULL_B_ITEMS, levels=PRED_LEVELS["B"])
    b.update(windows_fixed=bw, prof_windows=[], est_min=est, cap_min=cap, time_model=tm, d1_items=dict(FULL_B_ITEMS), title=TITLES["B"],
             e34=dict(PLAN["B"], seqs=int(seqs)), e34_smoke=False)
    arms = {BASE_KEY: b}
    for key in LEVERS:
        arms[key] = lever_arm(D1, N, b, key, sw, w if key in W_ARMS else int(seqs))
    return arms


def lever_arm(D1, N, b, key, switch=None, w=W_PRED):
    """One lever = a deep copy of B (an arm_plan B) + its change, with its items / windows / time. w = S16B2W's
    max_num_seqs (ignored on the other arms). Used at build time and by the run-time W re-plan."""
    sw = switch_map(switch)
    if key in W_ARMS:
        a = w_arm(D1, N, b, w)
    else:
        a = copy.deepcopy(b)
        argv = SSM_ARGS + (B2_ARGS if key in B2_ARMS else [])
        a["extra"] = list(b["extra"]) + argv
        env_add = {ENV_SW: SW_B2} if key in B2_ARMS else {}
        a["env"].update(env_add)
        a["coverage"] = N.coverage(a["captures"], a["streams"], a["k"])
        a["e34"] = dict(PLAN[key], seqs=int(b["streams"]))
        a["d1_change"] = {"kind": "e34", "env": env_add, "argv": argv, "captures": None}
        a["expect"] = D1.expect_of(N, a)
    smoke = key == "S16B2" and sw.get("SMOKE")
    if smoke:
        wins, items, rc = [], dict(SMOKE_ITEMS), "first"
    else:
        wins = copy.deepcopy(LEVER_WINDOWS) + (max_windows(a["streams"]) if key in W_ARMS else [])
        items, rc = dict(FULL_LEVER_ITEMS), READY_CLASS[key]
    lv = PRED_LEVELS.get(key)
    if key in W_ARMS and lv:
        lv = {bd: min(int(x), int(a["streams"])) for bd, x in lv.items()}
    est, cap, tm = arm_time(key, wins, items, ready_class=rc, levels=lv, smoke=smoke)
    a.update(key=key, name=f"e34_{key.lower()}", port=PORTS[key], title=TITLES[key] + (" | SMOKE plan" if smoke else ""), lever=True,
             windows_fixed=wins, prof_windows=[], est_min=est, cap_min=cap, time_model=tm, d1_items=items, base_change=None,
             e34_smoke=bool(smoke))
    return a


def neverdo(T, arm, env, argv):
    """nb4t_lib.never_do_violations minus exactly the lifted entry on an arm whose plan carries --mamba-ssm-cache-dtype
    bfloat16 (and whose argv's last value of the flag is bfloat16)."""
    v = list(T.never_do_violations(env, argv))
    if (arm.get("e34") or {}).get("ssm") == "bfloat16" and T.last_value(argv, SSM_FLAG) == "bfloat16":
        v = [x for x in v if x != LIFTED]
    return v


def check_arm_plan(D1, N, T, arms, d3s, switch=None):
    """Fail closed (ValueError): the order; B = NB-D1's B (argv, spec, captures, env, pin, seqs); every arm a DFlash2 arm
    with EMBED_HOST from the first try, Q-defer, the pin, no never-do item but the lifted one on an SSM arm; per arm the
    planned SSM flag / block flag / SW block / seqs / captures exactly; exact FULL graphs for every decode batch; 90k and
    116k windows on every non-smoke arm; the expect = the env; the full plan's oracle / long oracle / prefix / TF items."""
    errs = []
    sw = switch_map(switch)
    if tuple(arms) != ARM_KEYS:
        raise ValueError(f"E34_ARM_PLAN_INVALID: order {tuple(arms)} != {ARM_KEYS}")
    d1b = D1.arm_plan(N, d3s, switch=None)[BASE_KEY]
    b = arms[BASE_KEY]
    for f in ("extra", "spec", "captures", "k", "method", "q_label", "needs_patch", "kv_bytes", "env", "streams", "expect"):
        if b.get(f) != d1b.get(f):
            errs.append(f"B.{f} != NB-D1 B.{f}")
    base_extra, tail = D1._split_tail(b["extra"])
    names, ports = set(), set()
    for key, a in arms.items():
        names.add(a["name"])
        ports.add(a["port"])
        pl = PLAN[key]
        ex = a["extra"]
        v = neverdo(T, a, a["env"], ex)
        if v:
            errs.append(f"{key}: never-do {v}")
        if a.get("method") != "dflash" or a["env"].get(N.ENV_SEAM) != "1" or a.get("k") != 3:
            errs.append(f"{key}: not a DFlash2 K3 arm")
        if a["env"].get(D1.ENV_EH) != "1" or not a.get("d1_eh_first"):
            errs.append(f"{key}: DFLASH_DRAFT_EMBED_HOST=1 not from the first try")
        if a.get("parent") != "D3s" or not a.get("needs_patch") or a.get("q_label") != N.Q_LABEL or a["env"].get(T.ENV_DEFER) != "1":
            errs.append(f"{key}: parent / Q-defer / patched tree")
        if a["kv_bytes"] != b["kv_bytes"] or a["kv_bytes"] != PIN:
            errs.append(f"{key}: pin {a['kv_bytes']}")
        want_ssm = None if pl["ssm"] == "float32" else pl["ssm"]
        if T.last_value(ex, SSM_FLAG) != want_ssm or ex.count(SSM_FLAG) != (0 if want_ssm is None else 1):
            errs.append(f"{key}: {SSM_FLAG} {T.last_value(ex, SSM_FLAG)} != {want_ssm}")
        want_blk = None if pl["block"] == MERLIN_BLOCK else str(pl["block"])
        if T.last_value(ex, BLOCK_FLAG) != want_blk or ex.count(BLOCK_FLAG) != (0 if want_blk is None else 1):
            errs.append(f"{key}: {BLOCK_FLAG} {T.last_value(ex, BLOCK_FLAG)} != {want_blk} (Merlin's own 4096 otherwise)")
        if _sw_key(a) != str(pl["sw"]):
            errs.append(f"{key}: {ENV_SW} {_sw_key(a)} != {pl['sw']}")
        s = int((a.get("e34") or {}).get("seqs") or 0)
        if key not in W_ARMS and s != SEQS:
            errs.append(f"{key}: seqs {s}")
        if key in W_ARMS and not (W_MIN <= s <= W_MAX):
            errs.append(f"{key}: W {s} outside {W_MIN} .. {W_MAX}")
        if a["streams"] != s or a["expect_cfg"]["max_num_seqs"] != s or T.last_value(ex, SEQS_FLAG) != str(s):
            errs.append(f"{key}: --max-num-seqs {T.last_value(ex, SEQS_FLAG)} / streams {a['streams']} != {s}")
        if a["captures"] != D1.dflash_captures(s, 3) or N.captures_from_argv(ex) != a["captures"] or a["expect_cfg"].get("captures") != a["captures"]:
            errs.append(f"{key}: captures != every multiple of 4 up to 4 x {s}")
        if (T.spec_from_argv(ex) or {}) != b["spec"] or a["spec"] != b["spec"]:
            errs.append(f"{key}: spec != B's")
        cov = a.get("coverage") or {}
        if not cov.get("covers_all") or cov.get("padded_b"):
            errs.append(f"{key}: decode batches without an exact FULL graph {cov}")
        if set(a["expect"]) != set(N.WATCHED) | set(D1.D1_ENV_KEYS) or a["expect"] != D1.expect_of(N, a):
            errs.append(f"{key}: expect != the env (WATCHED + the NB-D1 keys)")
        if not a.get("e34_smoke"):
            if not any(w[0] == "90k" for w in a["windows_fixed"]) or not any(w[0] == "116k" for w in a["windows_fixed"]):
                errs.append(f"{key}: the window set must include 90k and 116k (user rule)")
            want_items = FULL_B_ITEMS if key == BASE_KEY else FULL_LEVER_ITEMS
            if a.get("d1_items") != want_items:
                errs.append(f"{key}: items {a.get('d1_items')} != the full plan")
        elif key != "S16B2" or not sw.get("SMOKE") or a.get("windows_fixed") != [] or a.get("d1_items") != SMOKE_ITEMS:
            errs.append(f"{key}: a smoke plan outside S16B2 / SMOKE")
        if key == BASE_KEY:
            continue
        diff = {x: a["env"].get(x) for x in set(a["env"]) | set(b["env"]) if a["env"].get(x) != b["env"].get(x)}
        if diff != ({ENV_SW: SW_B2} if key in B2_ARMS else {}):
            errs.append(f"{key}: env differs from B by {diff}")
        if key in W_ARMS:
            ok = ex == base_extra + tail[:2] + ["--compilation-config", N.compilation_json(a["captures"]), SEQS_FLAG, str(s)] + SSM_ARGS + B2_ARGS
        else:
            ok = ex == b["extra"] + SSM_ARGS + (B2_ARGS if key in B2_ARMS else []) and a["captures"] == b["captures"]
        if not ok:
            errs.append(f"{key}: argv is not B's + exactly its change")
    if len(names) != len(arms) or len(ports) != len(arms) or max(ports) + 8 > 8355 or min(ports) < 8000:
        errs.append("names / ports not distinct or outside Merlin's 8000..8355 scheme (+ retry offsets)")
    if errs:
        raise ValueError("E34_ARM_PLAN_INVALID: " + "; ".join(errs))
    return True


def launch_check(N, T, arm, env, argv, kv_bytes=None):
    """nbdf_lib.launch_check on the FULL final env / argv minus exactly the lifted never-do entry on a planned SSM arm, plus
    E34's own: the last --mamba-ssm-cache-dtype / --block-size / --max-num-seqs and the SW block env as planned."""
    bad = list(N.launch_check(arm, env, argv, kv_bytes))
    pl = arm.get("e34") or {}
    if pl.get("ssm") == "bfloat16" and T.last_value(argv, SSM_FLAG) == "bfloat16":
        bad = [x for x in bad if x != LIFTED]
    want_ssm = None if pl.get("ssm") in (None, "float32") else pl.get("ssm")
    if T.last_value(argv, SSM_FLAG) != want_ssm:
        bad.append(f"{arm.get('key')} {SSM_FLAG} {T.last_value(argv, SSM_FLAG)!r} != {want_ssm!r}")
    want_blk = str(pl.get("block") or MERLIN_BLOCK)
    if T.last_value(argv, BLOCK_FLAG) != want_blk:
        bad.append(f"{arm.get('key')} {BLOCK_FLAG} {T.last_value(argv, BLOCK_FLAG)!r} != {want_blk!r}")
    if str((env or {}).get(ENV_SW)) != str(pl.get("sw")):
        bad.append(f"{arm.get('key')} launch env {ENV_SW}={(env or {}).get(ENV_SW)!r} != {pl.get('sw')!r}")
    return bad


# ------------------------------------------------------------------ W from S16B2's measured pool (the boot log)
def f_blocks(pool, mc, block, max_len=131072):
    """= nbdf_lib.f_blocks (F from the max-concurrency line at the attention block)."""
    if not (isinstance(pool, (int, float)) and not isinstance(pool, bool) and pool > 0 and isinstance(mc, (int, float))
            and not isinstance(mc, bool) and math.isfinite(mc) and mc > 0 and isinstance(block, int) and block > 0):
        return None
    f = math.ceil(round(pool / mc - max_len / block, 3))
    return f if 0 <= f <= pool else None


def w_from_capacity(cap, band_tokens=W_BAND_TOKENS, lo=W_MIN, hi=W_MAX):
    """(W, info): the static admission bound at band_tokens = (pool - 1 null block - 1 QSA ring block) // (ceil(band /
    attention block) + F), clamped to lo .. hi. cap = e34_capacity's record of S16B2's boot. Unreadable -> (None, info)."""
    cap = cap or {}
    pool, mc, blk = cap.get("pool"), cap.get("mc_131072"), cap.get("attn_block")
    f = f_blocks(pool, mc, blk) if isinstance(blk, int) else None
    info = {"pool": pool, "mc_131072": mc, "attn_block": blk, "F": f, "band_tokens": band_tokens}
    if f is None:
        return None, dict(info, source="UNREADABLE")
    n = (int(pool) - 2) // (math.ceil(band_tokens / blk) + f)
    return max(lo, min(hi, int(n))), dict(info, raw=int(n), source="S16B2 boot: num_gpu_blocks + the max-concurrency line")


# ------------------------------------------------------------------ boot read-backs (capacity: records; dtype / block: real failures)
RX = {"group_sizes": re.compile(r"kv cache group sizes \(([^)]*)\)"),
      "pad_pct": re.compile(r"Padding mamba page size by ([\d.]+)%"),
      "set_block": re.compile(r"Setting attention block size to (\d+) tokens"),
      "kv_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
      "max_conc": re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"),
      "ssm_warn": re.compile(r"specifies mamba_ssm_dtype='(\w+)' in its config, but --mamba-ssm-cache-dtype='(\w+)' was passed"),
      "nondefault_ssm": re.compile(r"'mamba_ssm_cache_dtype': '(\w+)'"),
      "nondefault_block": re.compile(r"'block_size': (\d+)")}


def _nondefault(t):
    return next((l for l in (t or "").splitlines() if "non-default args:" in l), "")


def log_lines(t):
    """The boot log's capacity / dtype lines (recorded; the first group-size line, every padding %)."""
    t = t or ""
    gs = RX["group_sizes"].findall(t)
    mc = RX["max_conc"].findall(t)
    nd = _nondefault(t)
    sw_ = RX["ssm_warn"].findall(t)
    return {"kv_cache_group_sizes": [int(x) for x in gs[0].replace(" ", "").split(",") if x] if gs else None,
            "padding_mamba_pct": sorted({float(x) for x in RX["pad_pct"].findall(t)}),
            "setting_attention_block": sorted({int(x) for x in RX["set_block"].findall(t)}),
            "gpu_kv_cache_tokens": sorted({int(x.replace(",", "")) for x in RX["kv_tokens"].findall(t)}),
            "max_concurrency": sorted({(int(a.replace(",", "")), float(b)) for a, b in mc}),
            "ssm_override_warning": sorted(set(sw_)),
            "nondefault_mamba_ssm_cache_dtype": (RX["nondefault_ssm"].findall(nd) or [None])[-1],
            "nondefault_block_size": int((RX["nondefault_block"].findall(nd) or [0])[-1]) or None,
            "lines": [l.strip()[:260] for l in t.splitlines() if any(r.search(l) for k, r in RX.items() if k not in ("nondefault_ssm", "nondefault_block"))][:16]}


def worker_view(workers):
    """Per rank from the NB1 worker extension (nb1_cache_info): cache_config.mamba_ssm_cache_dtype / block_size /
    mamba_page_size_padded / num_gpu_blocks, the GDN spec's state dtypes + unpadded page, the drafter SW block, the
    kv_cache_config group blocks."""
    out = []
    for w in ((workers or {}).get("ranks") or []):
        if not isinstance(w, dict):
            continue
        gdn = [s for s in (w.get("kv_cache_specs") or []) if isinstance(s, dict) and s.get("type") == "MambaSpec"
               and "linear_attn" in str(s.get("example") or "")]
        swd = [s for s in (w.get("kv_cache_specs") or []) if isinstance(s, dict) and "Sliding" in str(s.get("type"))]
        att = [s for s in (w.get("kv_cache_specs") or []) if isinstance(s, dict) and s.get("type") == "FullAttentionSpec"]
        kc = w.get("kv_cache_config") or {}
        out.append({"rank": w.get("rank"), "mamba_ssm_cache_dtype": w.get("mamba_ssm_cache_dtype"), "cache_block_size": w.get("cache_block_size"),
                    "mamba_page_size_padded": w.get("mamba_page_size_padded"), "num_gpu_blocks": w.get("num_gpu_blocks"),
                    "gdn_state_dtypes": (gdn[0].get("state_dtypes") if gdn else None), "gdn_unpadded_page": (gdn[0].get("unpadded_page_bytes") if gdn else None),
                    "gdn_page": (gdn[0].get("page_size_bytes") if gdn else None), "sw_block": (swd[0].get("block_size") if swd else None),
                    "attn_block": (att[0].get("block_size") if att else None),
                    "group_blocks": [g.get("block_size") for g in kc.get("groups") or []]})
    return out


def readback(arm, log_text, workers, dflash_rb=None):
    """E34's real failures for THIS arm: the planned SSM dtype / attention block / drafter SW block read back with another
    value (worker cache_config, the GDN spec state dtype, the FullAttention spec block, the drafter SW spec / seam events, the
    platform's 'Setting attention block size' line). Unreadable -> recorded only."""
    pl = arm.get("e34") or {}
    wv = worker_view(workers)
    ll = log_lines(log_text)
    why, unread = [], []
    dts = [r["mamba_ssm_cache_dtype"] for r in wv if r.get("mamba_ssm_cache_dtype") not in (None, "None")]
    if not dts:
        unread.append("cache_config.mamba_ssm_cache_dtype (worker)")
    elif any(str(d) != pl.get("ssm") for d in dts):
        why.append(f"cache_config.mamba_ssm_cache_dtype {dts} != {pl.get('ssm')}")
    sd = [r["gdn_state_dtypes"][1] for r in wv if isinstance(r.get("gdn_state_dtypes"), list) and len(r["gdn_state_dtypes"]) > 1]
    if not sd:
        unread.append("GDN MambaSpec state dtypes")
    elif any(str(d).replace("torch.", "") != pl.get("ssm") for d in sd):
        why.append(f"GDN SSM state dtype {sd} != {pl.get('ssm')}")
    blk = [r["cache_block_size"] for r in wv if isinstance(r.get("cache_block_size"), int)]
    if not blk:
        unread.append("cache_config.block_size (worker)")
    elif any(x != pl.get("block") for x in blk):
        why.append(f"worker cache_config.block_size {blk} != {pl.get('block')}")
    ab = [r["attn_block"] for r in wv if isinstance(r.get("attn_block"), int)]
    if not ab:
        unread.append("FullAttentionSpec block")
    elif any(x != pl.get("block") for x in ab):
        why.append(f"FullAttentionSpec block {ab} != {pl.get('block')}")
    if any(x != pl.get("block") for x in ll["setting_attention_block"]):
        why.append(f"the platform raised the attention block: 'Setting attention block size to {ll['setting_attention_block']} tokens'")
    swb = [r["sw_block"] for r in wv if isinstance(r.get("sw_block"), int)]
    if not swb:
        unread.append("drafter SlidingWindowSpec block")
    elif any(x != pl.get("sw") for x in swb):
        why.append(f"drafter SW block {swb} != {pl.get('sw')}")
    chk = (dflash_rb or {}).get("checks") or {}
    if chk.get("draft_sw_block_applied") is False:
        why.append("seam draft_sw_block events != DFLASH_DRAFT_SW_BLOCK")
    rec = {"arm": arm.get("key"), "plan": dict(pl), "workers": wv, "log": ll, "unreadable": unread, "why": "; ".join(why) or None,
           "real_failure": bool(why), "ssm_dtype_per_rank": dts, "block_per_rank": blk, "sw_block_per_rank": swb,
           "draft_sw_block_applied": chk.get("draft_sw_block_applied")}
    return rec


def capacity(o, log_text, workers=None):
    """Records: pool, attention block, the max-concurrency line, F (blocks a request beyond its attention blocks, from that
    line) + its token-equivalent, the GPU KV cache size line, the group sizes, the padding %, the static residents at 40k /
    64k / 90k / 116k and the CPU planner's prediction."""
    cap = (o or {}).get("capacity_after_ready") or {}
    nb = cap.get("num_gpu_blocks") or []
    pool = nb[0] if len(nb) == 1 else None
    lay = ((o or {}).get("kv_layout") or {}).get("block_sizes") or {}
    ab = (lay.get("FullAttentionSpec") or [None])[0]
    if not isinstance(ab, int):
        wv = [r.get("attn_block") for r in worker_view(workers or (o or {}).get("workers")) if isinstance(r.get("attn_block"), int)]
        ab = wv[0] if wv else None
    mc = cap.get("max_concurrency_131072")
    ll = log_lines(log_text)
    f = f_blocks(pool, mc, ab) if isinstance(ab, int) else None
    res = {}
    for L in (40000, 64000, 90000, 116000):
        res[str(L)] = ((pool - 2) // (math.ceil(L / ab) + f)) if (f is not None and pool) else None
    return {"pool": pool, "attn_block": ab, "mc_131072": mc, "F": f, "F_token_equiv": (f * ab) if (f is not None and ab) else None,
            "kv_tokens": cap.get("kv_tokens"), "block_bytes_per_stage": cap.get("block_bytes_per_stage"),
            "pin_unused_mib_per_stage": cap.get("pin_unused_mib_per_stage"), "log": ll, "static_residents": res,
            "cpu_planner": CPU_PLANNER.get(((o or {}).get("arm") or "")), "sw_block_layout": lay.get("SlidingWindowSpec")}


def boot_line(minutes, key, cap, rb):
    def f(v):
        return "n/a" if v is None else str(v)
    rb = rb or {}
    return (f"[LIVE {minutes:.1f}] {key} boot | pool {f(cap.get('pool'))} x {f(cap.get('attn_block'))} | F {f(cap.get('F'))} "
            f"({f(cap.get('F_token_equiv'))} tok-eq) | mc@131k {f(cap.get('mc_131072'))} | KV tokens {f(cap.get('kv_tokens'))} | "
            f"pad {cap.get('log', {}).get('padding_mamba_pct')} % | groups {cap.get('log', {}).get('kv_cache_group_sizes')} | "
            f"ssm {sorted(set(map(str, rb.get('ssm_dtype_per_rank') or []))) or 'n/a'} | block {sorted(set(rb.get('block_per_rank') or [])) or 'n/a'} | "
            f"sw {sorted(set(rb.get('sw_block_per_rank') or [])) or 'n/a'} | readback {'FAIL: ' + str(rb.get('why'))[:120] if rb.get('real_failure') else 'ok'}")


# ------------------------------------------------------------------ the [LIVE] lines (print only)
def b_window_for(b_rec, wkey):
    """B's window for a lever window: the same key; a W window -> B's W window of the same band (max residency vs max
    residency); else the first valid same-band window."""
    st = (b_rec or {}).get("steady") or {}
    w = st.get(wkey)
    if isinstance(w, dict) and isinstance(w.get("tok_s"), (int, float)):
        return wkey, w
    band, tagw = str(wkey).split("@")[0], str(wkey).endswith("W")
    cands = [(k, v) for k, v in st.items() if isinstance(v, dict) and not str(k).startswith("profile") and str(k).split("@")[0] == band
             and isinstance(v.get("tok_s"), (int, float))]
    for k, v in cands:
        if str(k).endswith("W") == tagw:
            return k, v
    return cands[0] if cands else (None, None)


def live_line(D1, minutes, arm_key, wkey, w, b_rec=None):
    """D1.live_line (tok/s, AL, running, preempt, vs B) with E34's B-window match + the window's prefix hit rate."""
    w = w or {}
    bk, bw = (None, None) if arm_key == BASE_KEY else b_window_for(b_rec, wkey)
    fake_b = {"steady": {wkey: bw}} if bw is not None else None
    line = D1.live_line(minutes, arm_key, wkey, w, fake_b)
    ph = w.get("prefix_hit_rate")
    return line + f" | prefix hit {ph if ph is not None else 'n/a'}" + (f" | vs B window {bk}" if bk and bk != wkey else "")


def _st(x):
    return (x or {}).get("status") if isinstance(x, dict) else x


def item_line(minutes, arm_key, item, val):
    """One [LIVE] line per test (smoke, oracle, oracle A/A, long oracle, prefix test, TF ref / A/A / vs B, the boot)."""
    v = val if isinstance(val, dict) else {}
    if item == "smoke":
        acc = v.get("acceptance") or {}
        s = f"AL {acc.get('al')} | tokens {v.get('tokens')} | errors {v.get('errors')} | garbage {(v.get('garbage') or {}).get('hard') if isinstance(v.get('garbage'), dict) else v.get('garbage')}"
    elif item in ("oracle", "oracle_aa"):
        vs = {k: _st(x) for k, x in v.items() if str(k).startswith("vs_")}
        rows = [(r.get("first_diff"), r.get("oracle_margin")) for x in v.values() if isinstance(x, dict) for r in (x.get("rows") or [])][:6]
        s = f"{vs} | first diff / margin {rows} | errors {v.get('errors')}"
    elif item == "long_oracle":
        vs = {k: _st(x) for k, x in v.items() if str(k).startswith("vs_")}
        s = f"{vs} | tokens {v.get('tokens')} | prompt {v.get('prompt_tokens')} | errors {v.get('errors')} | garbage {v.get('garbage')}"
    elif item == "prefix_cache":
        rows = []
        for p in v.get("prompts") or []:
            for t_, tv in (p.get("turns") or {}).items():
                wm = (tv.get("warm") or {})
                rows.append((p.get("variant"), t_, (wm.get("prefix") or {}).get("rate"), _st(tv.get("warm_vs_cold"))))
        s = f"block {v.get('block')} | (variant, turn, warm hit rate, warm vs cold) {rows}"
    elif item in ("tf_ref", "tf_aa", "tf_vs_q"):
        s = " | ".join(f"{b} {x.get('status') or ''} n {x.get('n')} mean|dlp| {x.get('mean_abs_dlogprob')} rank1 {x.get('rank1_agreement')}"
                       + (f" err {str(x.get('error'))[:60]}" if x.get("error") else "") for b, x in v.items() if isinstance(x, dict))
    else:
        s = json.dumps(v, default=str)[:300]
    return f"[LIVE {minutes:.1f}] {arm_key} {item} | {s}"[:1500]


LIVE_ITEMS = ("smoke", "oracle", "oracle_aa", "long_oracle", "prefix_cache", "tf_ref", "tf_aa", "tf_vs_q")


# ------------------------------------------------------------------ TF traces (build time: the dataset file; run time: the set)
def tf_long_payload(traces):
    """traces = {name: (task_id, prompt_ids, p1_ids, meta)} -> the dataset JSON (prompt + p1 concatenated)."""
    out = {"what": "E34 teacher-forced long prompts: E31 direct Flash-Next recorded thinking traces (prompt + p1 token ids)", "traces": {}}
    for nm, (tid, pr, p1, meta) in traces.items():
        ids = [int(x) for x in pr] + [int(x) for x in p1]
        out["traces"][nm] = {"task": tid, "prompt_tokens": len(pr), "p1_tokens": len(p1), "len": len(ids), "meta": meta, "ids": ids}
    return out


def tf_set(prompts, tf_long, smoke=False):
    """The TF prompt set: smoke -> the 113k trace only; else RC2's 8k / 40k TF prompts + the three traces (a missing trace
    file -> the synthetic bands only, recorded)."""
    tl = ((tf_long or {}).get("traces") or {})
    if smoke:
        return {k: tl[k]["ids"] for k in TF_SMOKE if k in tl}
    out = {b: prompts[b][5] for b in TF_SYNTH}
    for k, _ in TF_LONG:
        if k in tl:
            out[k] = tl[k]["ids"]
    return out


# ------------------------------------------------------------------ records (record only)
def records(recs, arms, switch):
    """Per arm: the capacity read-back (pool, block, F, F tokens, mc, residents, padding, group sizes), the dtype / block
    read-back, the oracle / long oracle / TF / prefix numbers vs B, the windows (tok/s, AL, running, preemptions, prefix
    hit). Raw numbers; labels recorded only."""
    out = {"lib": LIB_VERSION, "switch": switch_map(switch), "arms": {}}
    for k in arms:
        r = recs.get(k) or {}
        st = r.get("steady") or {}
        out["arms"][k] = {
            "status": r.get("status"), "why": r.get("why") or r.get("early_stop"), "tries": r.get("tries"), "pin": r.get("pin"),
            "capacity": r.get("e34_capacity"), "readback": {x: (r.get("e34_readback") or {}).get(x) for x in
                                                            ("ssm_dtype_per_rank", "block_per_rank", "sw_block_per_rank", "why", "unreadable")},
            "w": r.get("e34_w"), "oracle_vs": r.get("oracle_vs"), "quality_gate": r.get("quality_gate"),
            "oracle_aa": ((r.get("oracle_aa") or {}).get("vs_q") or {}).get("status") if isinstance(r.get("oracle_aa"), dict) else None,
            "long_oracle_vs": {kk: _st(v) for kk, v in (r.get("long_oracle") or {}).items() if str(kk).startswith("vs_")},
            "long_gate": r.get("long_gate"), "prefix_gate": r.get("prefix_gate"),
            "tf": r.get("tf_vs_q") or r.get("tf_aa"), "tf_gate": r.get("tf_gate"), "tf_error": r.get("tf_error"),
            "smoke_gate": r.get("smoke_gate"),
            "windows": {wk: {f: (w or {}).get(f) for f in ("status", "tok_s", "mean_acceptance_length", "running_max", "level", "preemptions",
                                                           "prefix_hit_rate")} for wk, w in st.items() if isinstance(w, dict)}}
    return out


# ------------------------------------------------------------------ the E34 real-kit gate (build time)
def gate_from_report(base_gate, rep, closed_sha, keys=KIT_KEYS):
    g = copy.deepcopy(base_gate or {})
    why = []
    if not (base_gate or {}).get("passed"):
        why.append("the embedded NBDF-K100 real-kit gate did not pass")
    v = (rep or {}).get("verdict") or {}
    chk = v.get("checks") or {}
    nbs = (rep or {}).get("notebook_sha256")
    if not nbs or nbs != closed_sha:
        why.append(f"STALE_NOTEBOOK_SHA: the kit ran {nbs}, the closed E34 build is {closed_sha}")
    if v.get("ALL_OK") is not True:
        why.append(f"real kit not ALL_OK: {sorted(k for k, x in chk.items() if x is not True)[:12]}")
    miss = [k for k in keys if chk.get(k) is not True]
    if miss:
        why.append(f"E34 real-kit checks not True: {miss}")
    g.update(passed=not why, why=why or ["NBDF-K100 gate passed + the E34 real kit ALL OK on the closed build, provenance bound"],
             e34_kit_checks={k: chk.get(k) for k in keys}, e34_provenance={"kit_notebook_sha256": nbs, "closed_build_sha256": closed_sha})
    return g


def closed_gate(base_gate, why="E34 closed build: the E34 real CPU kit has not passed (build with --kv-gate-report)"):
    g = copy.deepcopy(base_gate or {})
    g.update(passed=False, why=[why], e34_kit_checks={k: None for k in KIT_KEYS}, e34_provenance=None)
    return g
