"""E34 unit tests (stdlib only; local Windows Python and the Py3.12 Kaggle CPU units kit).

  python e34_unit_tests.py [--libs DIR [DIR ...]] [--body FILE] [--facts FILE] [--export FILE] [--fixture FILE] [--notebook FILE]
                           [--capacity FILE]

Covers: the arm plan (B = NB-D1's B exactly; S16 / S16B2 / S16B2W = B + exactly their change; mutations refused; every W 24 .. 40
valid), the never-do lift (exactly 'argv --mamba-ssm-cache-dtype bfloat16' on the SSM arms only), E34.launch_check on the
launch env / argv as the notebook builds them (Merlin's own args incl. --block-size 4096 first, the arm's extra last), the W rule
on the boot capacity, the read-back (the NB-S1 GPU worker record = B; its S16 / S16B2 transforms; mismatches = real failures;
unreadable = recorded), the capacity record + boot line, the log parsers, the TF set, the time plan / budgets, the [LIVE] lines,
the kit gate, the body (markers, compile, KV gate vs facts), arm_body / arm_boot / nbd1_main / run_window with fakes, the facts /
export / copy block. Prints `FAIL <name>: <why>`, `SKIP <name>: <why>` and `<passed>/<total> PASS` last.
"""
import ast
import copy
import hashlib
import io
import json
import re
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")


def _arg(name, default=None, many=False):
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    if not many:
        return sys.argv[i + 1]
    out = []
    for x in sys.argv[i + 1:]:
        if x.startswith("--"):
            break
        out.append(x)
    return out


LIBS = _arg("--libs", [str(HERE), str(E25B), str(E27B)], many=True)
for d in reversed(LIBS):
    sys.path.insert(0, d)
sys.dont_write_bytecode = True
BODY = Path(_arg("--body", str(HERE / "e34_cell7_body.py")))
FACTS = Path(_arg("--facts", str(HERE / "e34_build_facts.json")))
EXPORT = Path(_arg("--export", str(HERE / "e34_code_export.py")))
FIXTURE = Path(_arg("--fixture", str(HERE / "e34_log_fixture.json")))
NBF = Path(_arg("--notebook", str(HERE.parent / "Experiment Notebooks" / "Q27-XHIGH-E34_FULL.ipynb")))
CAPF = Path(_arg("--capacity", str(HERE.parent / "Experiment Results" / "e34_capacity_probe.json")))
PIN = 7933526016

import nb4t_lib as T  # noqa: E402
import nbdf_lib as N  # noqa: E402
import nbd1_lib as D1  # noqa: E402
import e34_lib as E  # noqa: E402
try:
    import nb4_ppack as P  # noqa: E402
except Exception:
    P = None

RES = {"pass": 0, "total": 0}


def check(name, cond, why=""):
    RES["total"] += 1
    if cond:
        RES["pass"] += 1
    else:
        print(f"FAIL {name}: {why}"[:900], flush=True)


def skip(name, why):
    print(f"SKIP {name}: {why}", flush=True)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


# ------------------------------------------------------------------ the NBDF-K100 D3s (nbdf_lib configured as the K100 body)
K100_Q_CAPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96]
N.configure(SEQS=24, CONC_LADDER=(2, 8, 23), Q_CAPTURES=K100_Q_CAPS, Q_TAIL=["--compilation-config", N.compilation_json(K100_Q_CAPS)],
            WKEYS=("8k@16#1", "8k@16#2", "profile_8k@16", "40k@16"), PREFIX_RECORD_ONLY=True, SHORT_EXIT_ARMS=(),
            LONG_THR_FLOOR={"90k": 0.037, "116k": 0.054}, TF_MEAN_MARGIN=0.05, TF_RANK1_MARGIN=0.03, TF_REQUIRED=True)
Q_EXTRA = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}", "--max-num-seqs", "24"]
SEAM = {"DFLASH_SEAM": "1", "DFLASH_MASK_ROW_SHA": N.DRAFTER_PINS["mask_row"]["sha256"], "DFLASH_DRAFT_SW_BLOCK": "1024"}
QENV = P.server_env("Q-defer") if P else {T.ENV_DEFER: "1"}
ARMS0 = N.arm_plan(Q_EXTRA, {"VLLM_SERVER_DEV_MODE": "1", "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, PIN,
                   "/kaggle/input/models/foysalemonshanto/" + N.DRAFTER_REL, q_env=QENV, seam_envs={"D3s": dict(SEAM), "D4s": dict(SEAM)})
N.restrict_arms(("Q", "D3s", "D4s"))
N.check_arm_plan({k: ARMS0[k] for k in N.ORDER})
N.restrict_arms(("Q", "D3s"))
D3S = ARMS0["D3s"]
FX = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.is_file() else None


def plan(sw=None, w=E.W_PRED):
    return E.arm_plan(D1, N, D3S, seam_env_k4=dict(SEAM), switch=sw, w=w)


ARMS = plan(E.COPIES["full"])
MERLIN_ARGS = ["--tensor-parallel-size", "2", "--pipeline-parallel-size", "2", "--max-num-seqs", "8", "--max-num-batched-tokens", "1024",
               "--kv-cache-dtype", "fp8_e4m3", "--enable-prefix-caching", "--mamba-cache-mode", "align", "--block-size", "4096",
               "--use-replayssm", "--prefix-match-unit", "64"]


def launch_env(a):
    """Merlin build()'s plan env (partition + the QSA workspace at 256 / 256), then env.update(NB1_ENV[name]) = the arm's env."""
    env = {"VLLM_PP_LAYER_PARTITION": "25,23", "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "256", "VLLM_E2_QSA_LOGITS_GRANULE_MB": "256"}
    env.update(a["env"])
    return env


def launch_argv(a, kv=None):
    """vllm serve <ckpt> + Merlin's own args (its --block-size 4096) + the notebook's extra (last) + the pin."""
    return ["vllm", "serve", "/tmp/fn"] + MERLIN_ARGS + list(a["extra"]) + ["--kv-cache-memory-bytes", str(kv or a["kv_bytes"])]


# ------------------------------------------------------------------ 1. the arm plan
def t_plan():
    d1b = D1.arm_plan(N, D3S, switch=None)["B"]
    for nm, sw in list(E.COPIES.items()) + [("default", None)]:
        a = plan(sw)
        check(f"plan {nm}: order + valid", tuple(a) == E.ARM_KEYS and E.check_arm_plan(D1, N, T, a, D3S, sw) is True)
    b = ARMS["B"]
    check("plan B = NB-D1's B engine (argv / spec / captures / env / pin / seqs / expect)",
          all(b[f] == d1b[f] for f in ("extra", "spec", "captures", "env", "kv_bytes", "streams", "expect", "k", "method")))
    check("plan B: EMBED_HOST + QSA 128 / 64 + SW 1024, no SSM / block flag",
          b["env"]["DFLASH_DRAFT_EMBED_HOST"] == "1" and b["env"]["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] == "128"
          and b["env"]["VLLM_E2_QSA_LOGITS_GRANULE_MB"] == "64" and b["env"]["DFLASH_DRAFT_SW_BLOCK"] == "1024"
          and E.SSM_FLAG not in b["extra"] and E.BLOCK_FLAG not in b["extra"])
    s = ARMS["S16"]
    check("plan S16 = B + --mamba-ssm-cache-dtype bfloat16 only", s["extra"] == b["extra"] + ["--mamba-ssm-cache-dtype", "bfloat16"]
          and s["env"] == b["env"] and s["captures"] == b["captures"] and s["streams"] == 24)
    s2 = ARMS["S16B2"]
    check("plan S16B2 = B + SSM bf16 + --block-size 2048 + DFLASH_DRAFT_SW_BLOCK=512",
          s2["extra"] == b["extra"] + ["--mamba-ssm-cache-dtype", "bfloat16", "--block-size", "2048"]
          and s2["env"] == dict(b["env"], DFLASH_DRAFT_SW_BLOCK="512") and s2["expect"]["DFLASH_DRAFT_SW_BLOCK"] == "512")
    w = ARMS["S16B2W"]
    check("plan S16B2W: seqs W_PRED, captures 4 .. 4 W, SSM + block + SW 512", w["streams"] == E.W_PRED == 29
          and w["captures"] == [4 * i for i in range(1, 30)] and T.last_value(w["extra"], "--max-num-seqs") == "29"
          and T.last_value(w["extra"], "--block-size") == "2048" and T.last_value(w["extra"], "--mamba-ssm-cache-dtype") == "bfloat16"
          and w["env"]["DFLASH_DRAFT_SW_BLOCK"] == "512" and w["coverage"]["covers_all"])
    check("plan windows: 90k + 116k on every full arm; W windows on S16B2W; B max-resident (BMAX)",
          all(any(x[0] == "90k" for x in a["windows_fixed"]) and any(x[0] == "116k" for x in a["windows_fixed"]) for a in ARMS.values())
          and [x for x in w["windows_fixed"] if x[2] == "W"] == [["20k", 29, "W"], ["40k", 29, "W"], ["90k", 29, "W"], ["116k", 29, "W"]]
          and [x[0] for x in b["windows_fixed"] if x[2] == "W"] == ["40k", "90k", "116k"]
          and [x[:2] for x in ARMS["S16"]["windows_fixed"]] == [["8k", 16], ["20k", 24], ["40k", 16], ["90k", 8], ["116k", 6]])
    check("plan items: oracle / long oracle / prefix / TF on every full arm; the prefill probe on B only",
          all(a["d1_items"]["oracle"] and a["d1_items"]["long_oracle"] and a["d1_items"]["prefix"] and a["d1_items"]["tf"]
              and not a["d1_items"]["ladder"] for a in ARMS.values()) and b["d1_items"]["prefill_probe"]
          and not any(ARMS[k]["d1_items"]["prefill_probe"] for k in E.LEVERS))
    sm = plan(E.COPIES["smoke"])
    check("plan SMOKE: S16B2 = no windows, long oracle + TF, no oracle / prefix; first-boot ready",
          sm["S16B2"]["windows_fixed"] == [] and sm["S16B2"]["d1_items"] == E.SMOKE_ITEMS and sm["S16B2"]["e34_smoke"]
          and sm["S16B2"]["time_model"]["ready_class"] == "first" and not sm["S16"]["e34_smoke"])
    for ww in range(E.W_MIN, E.W_MAX + 1):
        a = dict(ARMS)
        a["S16B2W"] = E.lever_arm(D1, N, ARMS["B"], "S16B2W", E.COPIES["full"], ww)
        ok = E.check_arm_plan(D1, N, T, a, D3S, E.COPIES["full"]) is True and a["S16B2W"]["streams"] == ww
        if not ok:
            check(f"plan W {ww}", False)
    check("plan W 24 .. 40 all valid (the run-time re-plan range)", True)
    for nm, mut in (("B with SSM bf16", lambda a: a["B"]["extra"].extend(["--mamba-ssm-cache-dtype", "bfloat16"])),
                    ("S16 without its flag", lambda a: a["S16"]["extra"].__delitem__(slice(-2, None))),
                    ("S16 with block 2048", lambda a: a["S16"]["extra"].extend(["--block-size", "2048"])),
                    ("S16B2 SW 1024", lambda a: a["S16B2"]["env"].__setitem__("DFLASH_DRAFT_SW_BLOCK", "1024")),
                    ("S16B2 block 4096", lambda a: a["S16B2"]["extra"].__setitem__(-1, "4096")),
                    ("S16B2 SSM float16", lambda a: a["S16B2"]["extra"].__setitem__(-3, "float16")),
                    ("S16 another never-do item", lambda a: a["S16"]["extra"].append("--no-enable-prefix-caching")),
                    ("S16B2W W 41", lambda a: a.__setitem__("S16B2W", E.lever_arm(D1, N, a["B"], "S16B2W", None, 41))),
                    ("S16B2W captures to 96 only", lambda a: a["S16B2W"].__setitem__("captures", list(ARMS["B"]["captures"]))),
                    ("S16 seqs 28", lambda a: a["S16"]["extra"].extend(["--max-num-seqs", "28"])),
                    ("S16 pin", lambda a: a["S16"].__setitem__("kv_bytes", 7665090560)),
                    ("S16 no EMBED_HOST", lambda a: a["S16"]["env"].pop("DFLASH_DRAFT_EMBED_HOST")),
                    ("S16 no TF", lambda a: a["S16"]["d1_items"].__setitem__("tf", False)),
                    ("S16 no 116k window", lambda a: a["S16"].__setitem__("windows_fixed", [x for x in a["S16"]["windows_fixed"] if x[0] != "116k"])),
                    ("B QSA 256", lambda a: a["B"]["env"].__setitem__("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "256")),
                    ("order", lambda a: a.__setitem__("B", a.pop("B")))):
        a = copy.deepcopy(ARMS)
        mut(a)
        check(f"plan mutation refused: {nm}", raises(lambda: E.check_arm_plan(D1, N, T, a, D3S, E.COPIES["full"]), ValueError))
    a = copy.deepcopy(plan(E.COPIES["smoke"]))
    check("plan mutation refused: a smoke plan without SMOKE on", raises(lambda: E.check_arm_plan(D1, N, T, a, D3S, E.COPIES["full"]), ValueError))


# ------------------------------------------------------------------ 2. never-do lift + launch_check
def t_launch():
    check("nb4t_lib still lists the SSM bf16 never-do entry (lifted here, not edited)", T.NEVER_DO_ARG_VALUES.get("--mamba-ssm-cache-dtype") == ("bfloat16",))
    for k, a in ARMS.items():
        env, argv = launch_env(a), launch_argv(a)
        raw = N.launch_check(a, env, argv, a["kv_bytes"])
        check(f"launch {k}: nbdf_lib flags exactly {'the lifted entry' if k in E.SSM_ARMS else 'nothing'}",
              raw == ([E.LIFTED] if k in E.SSM_ARMS else []), raw)
        check(f"launch {k}: E34.launch_check passes", E.launch_check(N, T, a, env, argv, a["kv_bytes"]) == [], E.launch_check(N, T, a, env, argv, a["kv_bytes"]))
    b = ARMS["B"]
    for nm, a, env, argv, want in (
            ("B with SSM bf16 (never-do not lifted on B)", b, launch_env(b), launch_argv(b) + ["--mamba-ssm-cache-dtype", "bfloat16"], E.LIFTED),
            ("S16 without the flag", ARMS["S16"], launch_env(ARMS["S16"]), launch_argv(b), "--mamba-ssm-cache-dtype None"),
            ("S16B2 with SW 1024 in the env", ARMS["S16B2"], dict(launch_env(ARMS["S16B2"]), DFLASH_DRAFT_SW_BLOCK="1024"), launch_argv(ARMS["S16B2"]), "DFLASH_DRAFT_SW_BLOCK"),
            ("S16B2 with Merlin's block only", ARMS["S16B2"], launch_env(ARMS["S16B2"]), launch_argv(ARMS["S16B2"])[:-4] + ["--kv-cache-memory-bytes", str(PIN)], "--block-size"),
            ("S16 + another never-do item", ARMS["S16"], launch_env(ARMS["S16"]), launch_argv(ARMS["S16"]) + ["--no-enable-prefix-caching"], "argv --no-enable-prefix-caching"),
            ("S16B2W seqs 24", ARMS["S16B2W"], launch_env(ARMS["S16B2W"]), launch_argv(ARMS["S16B2W"]) + ["--max-num-seqs", "24"], "--max-num-seqs")):
        bad = E.launch_check(N, T, a, env, argv, PIN)
        check(f"launch negative: {nm}", any(want in x for x in bad), bad)
    check("neverdo: removes only the exact string, only on an SSM arm",
          E.neverdo(T, ARMS["S16"], {}, ["--mamba-ssm-cache-dtype", "bfloat16"]) == []
          and E.neverdo(T, ARMS["B"], {}, ["--mamba-ssm-cache-dtype", "bfloat16"]) == [E.LIFTED]
          and E.neverdo(T, ARMS["S16"], {}, ["--mamba-ssm-cache-dtype", "bfloat16", "--no-async-scheduling"]) == ["argv --no-async-scheduling"])


# ------------------------------------------------------------------ 3. W, capacity, read-back, logs
def _s16b2_workers(w, ssm="bfloat16", block=2048, sw=512):
    """B's GPU worker record transformed to what S16B2 must read back (the CPU planner's numbers)."""
    w = copy.deepcopy(w)
    for r in w["ranks"]:
        r["mamba_ssm_cache_dtype"], r["cache_block_size"] = ssm, block
        for s in r.get("kv_cache_specs") or []:
            if s["type"] == "MambaSpec" and "linear_attn" in s["example"]:
                s["state_dtypes"] = [s["state_dtypes"][0], ssm, s["state_dtypes"][2]]
                s["unpadded_page_bytes"] = E.GDN_STATE_BYTES.get(ssm)
            if s["type"] in ("MambaSpec", "FullAttentionSpec", "MLAAttentionSpec") and s["block_size"] == 4096:
                s["block_size"] = block
            if s["type"] == "SlidingWindowSpec":
                s["block_size"] = sw
    return w


def t_capacity_readback():
    check("W from the CPU planner numbers (1080, 13.320987654, 2048) = 29", E.w_from_capacity({"pool": 1080, "mc_131072": 1080 / 81.0, "attn_block": 2048})[0] == 29)
    w, info = E.w_from_capacity({"pool": 1080, "mc_131072": 13.32, "attn_block": 2048})
    check("W from the log's 2-decimal mc 13.32 -> F 18 -> 28", w == 28 and info["F"] == 18, info)
    check("W unreadable -> None", E.w_from_capacity({"pool": None, "mc_131072": 13.32, "attn_block": 2048})[0] is None
          and E.w_from_capacity({})[0] is None)
    check("W clamps to 24 .. 40", E.w_from_capacity({"pool": 400, "mc_131072": 400 / 81.0, "attn_block": 2048})[0] == 24
          and E.w_from_capacity({"pool": 4000, "mc_131072": 4000 / 81.0, "attn_block": 2048})[0] == 40)
    if FX is None:
        skip("readback", f"no fixture {FIXTURE}")
        return
    log_b = "\n".join(FX["log_lines"])
    o = {"arm": "B", "capacity_after_ready": FX["capacity_after_ready"], "kv_layout": FX["kv_layout"], "workers": FX["workers"]}
    cap = E.capacity(o, log_b, FX["workers"])
    check("capacity B on the NB-S1 GPU record: pool 540 x 4096, mc 12.25, F 13 (53,248 tok-eq), groups, padding 21.01 %",
          cap["pool"] == 540 and cap["attn_block"] == 4096 and cap["mc_131072"] == 12.25 and cap["F"] == 13 and cap["F_token_equiv"] == 53248
          and cap["log"]["kv_cache_group_sizes"] == [4096, 4096, 4096, 8, 4096, 1024] and cap["log"]["padding_mamba_pct"] == [21.01]
          and cap["log"]["gpu_kv_cache_tokens"] == [1605632] and cap["log"]["max_concurrency"] == [(131072, 12.25)], cap)
    rb = E.readback(ARMS["B"], log_b, FX["workers"], {"checks": FX["dflash_checks"]})
    check("readback B on the GPU record: fp32 / 4096 / SW 1024 on every rank, no failure",
          not rb["real_failure"] and set(rb["ssm_dtype_per_rank"]) == {"float32"} and set(rb["block_per_rank"]) == {4096}
          and set(rb["sw_block_per_rank"]) == {1024} and not rb["unreadable"], rb.get("why") or rb.get("unreadable"))
    rb = E.readback(ARMS["S16"], log_b, FX["workers"], {"checks": FX["dflash_checks"]})
    check("readback S16 on an fp32 boot: a real failure (the dtype not engaged)", rb["real_failure"] and "mamba_ssm_cache_dtype" in rb["why"], rb.get("why"))
    w16 = _s16b2_workers(FX["workers"], block=4096, sw=1024)
    check("readback S16 on its expected record: no failure", not E.readback(ARMS["S16"], log_b, w16, {"checks": FX["dflash_checks"]})["real_failure"])
    w2 = _s16b2_workers(FX["workers"])
    log2 = log_b.replace("21.01%", "10.78%").replace("(4096, 4096, 4096, 8, 4096, 1024)", "(2048, 2048, 2048, 8, 2048, 512)")
    rb = E.readback(ARMS["S16B2"], log2, w2, {"checks": dict(FX["dflash_checks"], draft_sw_block_applied=True)})
    check("readback S16B2 on its expected record: no failure", not rb["real_failure"], rb.get("why"))
    for nm, wk, lg, chk in (("the worker block raised to 4096", _s16b2_workers(FX["workers"], block=4096), log2, {}),
                            ("the platform's 'Setting attention block size to 4096'", w2,
                             log2 + "\n(Worker) INFO [interface.py:1099] Setting attention block size to 4096 tokens to ensure that attention page size is >= mamba page size.", {}),
                            ("the drafter SW block 1024", _s16b2_workers(FX["workers"], sw=1024), log2, {}),
                            ("the seam's draft_sw_block events disagree", w2, log2, {"draft_sw_block_applied": False}),
                            ("fp32 SSM", _s16b2_workers(FX["workers"], ssm="float32"), log2, {})):
        rb = E.readback(ARMS["S16B2"], lg, wk, {"checks": chk})
        check(f"readback S16B2 real failure: {nm}", rb["real_failure"], rb.get("why"))
    rb = E.readback(ARMS["S16B2"], "", {"ranks": []}, None)
    check("readback unreadable: recorded, not a failure", not rb["real_failure"] and len(rb["unreadable"]) >= 4, rb)
    o2 = {"arm": "S16B2", "capacity_after_ready": {"num_gpu_blocks": [1080], "max_concurrency_131072": 13.32, "kv_tokens": 2211840},
          "kv_layout": {"block_sizes": {"FullAttentionSpec": [2048], "SlidingWindowSpec": [512]}}}
    cap2 = E.capacity(o2, log2, w2)
    check("capacity S16B2 (planner numbers): pool 1080 x 2048, F 18 (36,864 tok-eq), residents 40k 28",
          cap2["pool"] == 1080 and cap2["attn_block"] == 2048 and cap2["F"] == 18 and cap2["F_token_equiv"] == 36864
          and cap2["static_residents"]["40000"] == 28 and cap2["cpu_planner"]["pool"] == 1080, cap2)
    ln = E.boot_line(12.3, "S16B2", cap2, E.readback(ARMS["S16B2"], log2, w2, {}))
    check("boot line: pool / F / tok-eq / mc / pad / groups / ssm / block / sw / readback ok",
          ln.startswith("[LIVE 12.3] S16B2 boot | pool 1080 x 2048 | F 18 (36864 tok-eq)") and "ssm ['bfloat16']" in ln and "block [2048]" in ln
          and "sw [512]" in ln and ln.endswith("readback ok") and "pad [10.78] %" in ln, ln)
    ll = E.log_lines("(APIServer) INFO non-default args: {'model_tag': '/x', 'mamba_ssm_cache_dtype': 'bfloat16', 'block_size': 2048}\n"
                     "(EngineCore) WARNING Qwen3.5 model specifies mamba_ssm_dtype='float32' in its config, but --mamba-ssm-cache-dtype='bfloat16' was passed. Using the user-specified value.")
    check("log lines: the non-default dtype / block + the override warning", ll["nondefault_mamba_ssm_cache_dtype"] == "bfloat16"
          and ll["nondefault_block_size"] == 2048 and ll["ssm_override_warning"] == [("float32", "bfloat16")], ll)
    if CAPF.is_file():
        cp = {r["e34_arm"]["name"].split(":")[0]: r for r in json.loads(CAPF.read_text(encoding="utf-8"))["results"]}
        check("CPU planner file = E34.CPU_PLANNER (pool / stride / F / mc)",
              all(min(cp[k]["num_blocks"]) == E.CPU_PLANNER[k]["pool"] and cp[k]["stride_per_rank"] == E.CPU_PLANNER[k]["stride"]
                  and cp[k]["F_admission"] == E.CPU_PLANNER[k]["F_adm"] and abs(cp[k]["max_concurrency_131072"]["24"] - E.CPU_PLANNER[k]["mc_131072"]) < 0.01
                  for k in ("B", "S16", "S16B2")) and min(cp["S16B2 (SW1024)"]["num_blocks"]) == 756
              and any(r.get("BLOCK_RAISED") for r in cp.values()))
        check("CPU planner B = the NB-S1 GPU boot (pool 540, mc 12.25)", min(cp["B"]["num_blocks"]) == FX["capacity_after_ready"]["num_gpu_blocks"][0]
              and cp["B"]["max_concurrency_131072"]["24"] == FX["capacity_after_ready"]["max_concurrency_131072"])
    else:
        skip("capacity_probe", f"no {CAPF}")


# ------------------------------------------------------------------ 4. TF set, time, live lines, gate
def t_misc():
    pr = {b: [[i] * 3 for i in range(8)] for b in ("8k", "40k")}
    tl = {"traces": {k: {"ids": [1, 2, 3]} for k, _ in E.TF_LONG}}
    check("tf_set full: 8k / 40k variant 5 + the three traces", list(E.tf_set(pr, tl)) == ["8k", "40k", "tf69k", "tf94k", "tf113k"]
          and E.tf_set(pr, tl)["8k"] == pr["8k"][5])
    check("tf_set smoke: the 113k trace only", list(E.tf_set(pr, tl, smoke=True)) == ["tf113k"])
    check("tf_set without the trace dataset: the synthetic bands only", list(E.tf_set(pr, None)) == ["8k", "40k"] and E.tf_set(pr, None, smoke=True) == {})
    pl = E.tf_long_payload({"x": ("t1", [1, 2], [3], {"m": 1})})
    check("tf_long_payload: prompt + p1", pl["traces"]["x"]["ids"] == [1, 2, 3] and pl["traces"]["x"]["len"] == 3)
    for x, sw in E.COPIES.items():
        a = plan(sw)
        bud = E.budget_for(a, sw)
        wk = E.copy_walk(a, sw, bud)
        check(f"time {x}: every arm fits its budget {bud} min, cap >= body + 1", wk["all_run"] and all(v["cap_min"] >= v["time_model"]["body"] + 1 for v in a.values()), wk)
    check("time FULL ~4 h (< 11 h: one copy), SMOKE <= 40 min", E.budget_for(plan(E.COPIES["full"]), E.COPIES["full"]) < 11 * 60
          and E.budget_for(plan(E.COPIES["smoke"]), E.COPIES["smoke"]) <= 40)
    b_rec = {"steady": {"40k@16": {"tok_s": 600.0}, "40k@20W": {"tok_s": 640.0}, "8k@16#1": {"tok_s": 650.0}}}
    check("b_window_for: a W window -> B's W window of the band", E.b_window_for(b_rec, "40k@22W")[0] == "40k@20W"
          and E.b_window_for(b_rec, "40k@16")[0] == "40k@16" and E.b_window_for(b_rec, "90k@8") == (None, None))
    ln = E.live_line(D1, 33.0, "S16B2W", "40k@22W", {"tok_s": 704.0, "mean_acceptance_length": 3.1, "running_max": 22, "preemptions": 0,
                                                     "prefix_hit_rate": 0.01}, b_rec)
    check("live line: tok/s, AL, running, preempt, vs B's W window, prefix hit", "tok/s 704.0" in ln and "AL 3.10" in ln and "running 22" in ln
          and "vs B +10.0 %" in ln and "prefix hit 0.01" in ln and "vs B window 40k@20W" in ln, ln)
    for item, val in (("smoke", {"acceptance": {"al": 3.1}, "tokens": [2048, 2048], "errors": []}),
                      ("oracle", {"vs_q": {"status": "EXACT", "rows": [{"first_diff": None, "oracle_margin": None}]}, "errors": [None]}),
                      ("long_oracle", {"vs_q_long": {"status": "NEAR_TIE"}, "tokens": [256, 256]}),
                      ("prefix_cache", {"block": 512, "prompts": [{"variant": 40, "turns": {"2": {"warm": {"prefix": {"rate": 0.38}}, "warm_vs_cold": {"status": "EXACT"}}}}]}),
                      ("tf_vs_q", {"tf113k": {"status": "MEASURED", "n": 256, "mean_abs_dlogprob": 0.02, "rank1_agreement": 0.99}})):
        s = E.item_line(5.0, "S16", item, val)
        check(f"item line {item}", s.startswith(f"[LIVE 5.0] S16 {item} | ") and len(s) < 1600, s)
    base = {"passed": True, "per_k": {"D3s": True}}
    rep = {"notebook_sha256": "abc", "verdict": {"ALL_OK": True, "checks": {k: True for k in E.KIT_KEYS}}}
    check("gate_from_report: ALL_OK + every key + provenance -> passed", E.gate_from_report(base, rep, "abc")["passed"] is True)
    check("gate_from_report: stale sha / a key False / base closed -> closed",
          E.gate_from_report(base, rep, "xyz")["passed"] is False
          and E.gate_from_report(base, dict(rep, verdict={"ALL_OK": True, "checks": dict(rep["verdict"]["checks"], e34_kv_probes_ok=False)}), "abc")["passed"] is False
          and E.gate_from_report({"passed": False}, rep, "abc")["passed"] is False)
    check("closed_gate", E.closed_gate(base)["passed"] is False and set(E.closed_gate(base)["e34_kit_checks"]) == set(E.KIT_KEYS))
    check("switch_map: absent -> default; a dict -> missing keys off", E.switch_map(None) == E.SWITCH_DEFAULT
          and E.switch_map({"S16B2": True}) == {k: k == "S16B2" for k in E.SWITCH_KEYS})


# ------------------------------------------------------------------ 5. the body: static checks + fakes
BTXT = BODY.read_text(encoding="utf-8") if BODY.is_file() else None
FNS = ({n.name: ast.get_source_segment(BTXT, n) for n in ast.parse(BTXT).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
       if BTXT else {})
EXTXT = EXPORT.read_text(encoding="utf-8") if EXPORT.is_file() else None
_m = re.search(r'K100_CFG = json.loads\(r"""(.*?)"""\)', EXTXT or "")
K100CFG = json.loads(_m.group(1)) if _m else None


def t_body_static():
    if not BTXT:
        skip("body", f"no body {BODY}")
        return
    b = BTXT
    check("body_no_cr", "\r" not in b)
    check("body_compiles", compile(b, "e34_cell7_body.py", "exec") is not None)
    for marker in ('E34.check_arm_plan(D1, N, T, ARMS, NBDF_ARMS["D3s"], NBD1_SWITCH)', 'globals().get("E34_ON")',
                   'bad = E34.launch_check(N, T, arm, s.env, s.argv, kv_bytes)', '        e34_capacity_rec(o, arm, t)                           # E34', '    e34_item_live(o, key, val)                    # E34',
                   'o["e34_readback"] = E34.readback(arm, t, o.get("workers"), o.get("dflash_readback"))',
                   'return "E34 read-back: "', 'a = e34_w_replan(k, a)', 'if is_q or arm.get("e34_smoke"):',
                   'arm["windows_fixed"] if arm.get("windows_fixed") is not None else K100_CFG["windows"]',
                   'if due(150) and arm.get("windows_fixed") != []:', 'print(E34.live_line(D1,', 'TF_P = E34.tf_set(PROMPTS, E34_TF_LONG',
                   'v["e34"] = dict(E34.records(', 'E34_LIB_SRC = _nbd1_ds("e34_lib.py", E34_LIB_SHA)', 'NBD1_SWITCH = E34.switch_map(_e34_on)',
                   '"planned_walk": E34.copy_walk(ARMS, NBD1_SWITCH, BUDGET / 60)}', 'oom = bool(o["oom_in_log"]) and not o["xid_in_log"]',
                   'nbd1_main()'):
        check(f"body_marker {marker[:70]}", b.count(marker) == 1, b.count(marker))
    check("body: NBDF's raw launch check no longer called in the launch path", "bad = N.launch_check(arm, s.env" not in b)
    check("body: the oracle / long oracle / prefix / TF code is NB-D1's (item switches, no new copies)",
          b.count("def long_oracle(") == 1 and b.count("def prefix_test(") == 1 and 'D1.item_on(arm, "tf")' in b and 'D1.item_on(arm, "prefix")' in b)
    m = re.search(r'^KV_KIT_GATE = json.loads\(r"""(.*?)"""\)', b, re.M)
    g = json.loads(m.group(1)) if m else {}
    fx = json.loads(FACTS.read_text(encoding="utf-8")) if FACTS.is_file() else {}
    check("body_kv_gate_matches_facts", m is not None and g.get("passed") == fx.get("gpu_ready") and "e34_kit_checks" in g, g.get("why"))
    i_s, i_p = b.find("E34 = importlib.import_module"), b.find("TF_P = E34.tf_set(")
    check("body: E34 setup before the TF set", 0 < i_s < i_p)


class Rec:
    def __init__(self, name, log, ret=None):
        self.name, self.log, self.ret = name, log, ret

    def __call__(self, *a, **k):
        self.log.append((self.name, a, k))
        r = self.ret(*a, **k) if callable(self.ret) else self.ret
        return {} if r is None else r


class Mod:
    def __init__(self, name, log, spec):
        self._n, self._log, self._spec = name, log, spec

    def __getattr__(self, a):
        if a.startswith("__"):
            raise AttributeError(a)
        v = self._spec.get(a, None)
        if a in self._spec and not callable(v):
            return v
        return Rec(f"{self._n}.{a}", self._log, v)


class _Srv:
    port, name, argv = 8011, "srv", []

    def alive(self):
        return True

    def log(self):
        return ""


class _W:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def body_run(arm, switch, smoke_gate="INCONCLUSIVE_SMOKE_INCONCLUSIVE_LADDER", d1f=None, measured=True):
    log = []
    nspec = {"FINE_GRAINED_RX": "zz-fine", "DRAFT_GROUP_WARN_RX": "zz-group", "LONG_BANDS": ("90k", "116k"), "BLOCK_TOKENS": 4096,
             "ADMISSION_PROBE_ARMS": (), "smoke_gate": lambda *a, **k: smoke_gate, "pos0_label": lambda *a, **k: {"label": "OK"},
             "spec_mismatch": lambda *a, **k: (False, ""), "qdefer_real_failure": lambda *a, **k: (False, "ok"),
             "k100_rel": lambda s, c: s, "contamination_gate": lambda *a: "INCONCLUSIVE_LADDER", "q_aa_clean": lambda *a: False,
             "long_oracle_gate": lambda *a: "PASS", "prefix_gate": lambda *a: "PASS", "engine_block": lambda *a: 4096,
             "spec_decoding_lines": lambda *a: [], "kv_bound": lambda *a: False, "tf_gate": lambda *a: {"status": "PASS"}}
    ns = {"N": Mod("N", log, nspec), "T": Mod("T", log, {}), "P": Mod("P", log, {"parse_log": lambda *a: []}),
          "L": Mod("L", log, {}), "D1": D1, "E34": E, "json": json, "re": re, "time": time, "traceback": traceback,
          "FAKE": False, "CUR": {"kv_bytes": PIN}, "LIVE": {}, "KV_KIT_GATE": {}, "Q_AA": {}, "Q_ACC": {}, "Q_TF_AA": {},
          "PREFILL": {"tok_s": 4000.0, "measured": measured}, "K100_CFG": K100CFG, "NBD1_SWITCH": E.switch_map(switch),
          "RESP": {"counts": {}, "arm": None}, "MEM_RX": re.compile(r"Model loading took"), "_watched": _W, "ROWS": {}}

    def _live(o, k, v):
        o[k] = v
    for nm, ret in (("due", True), ("short_gate_failed", False), ("_mem_gate", (True, "ok")), ("quality_gate", "PASS"),
                    ("band_level", lambda o, b: (99, {"source": "fake"})), ("boot_facts", lambda s, o, n: o),
                    ("jit_kernels", lambda *a: []), ("smoke", lambda p, a: {"acceptance": {}, "ladder_on": D1.item_on(a, "ladder")}),
                    ("run_window", lambda o, s, a, b, n, tag="": f"{b}@{n}{tag}"), ("profile_window", {"status": "DONE"}),
                    ("i8_readback", {"label": "I8_ENGAGED"}), ("k100_boot_gates", {"real_failure": False}),
                    ("win_need", 10.0), ("d1_readback_failure", d1f), ("d1_records", {"rec": True}),
                    ("oracle", {"vs_q": {"status": "EXACT"}}), ("long_oracle", {"vs_q_long": {"status": "EXACT"}}),
                    ("prefix_test", {"prompts": []}), ("tforced", {"8k": {"n": 256}}), ("e34_capacity_rec", None),
                    ("sibling_probe", {"label": {"label": "x"}}),
                    ("dflash_rb", {"checks": {"draft_load_check": True, "embed_host_as_planned": True}})):
        ns[nm] = Rec(nm, log, ret)
    ns["_live"] = _live
    free = set()
    for node in ast.walk(ast.parse(FNS["arm_body"])):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            free.add(node.id)
    for nm in free:
        if nm not in ns and nm not in dir(__import__("builtins")):
            ns[nm] = Rec(nm, log)
    exec(FNS["arm_body"], ns)
    o = ns["arm_body"](arm, "k100_none" if arm["key"] == "B" else "q", "q" if arm["key"] == "B" else None)([_Srv()])
    return o, [c[0] for c in log], log


def windows_called(log):
    return [f"{c[1][3]}@{c[1][4]}{c[1][5] if len(c[1]) > 5 else ''}" for c in log if c[0] == "run_window"]


def t_body_fakes():
    if not FNS or not K100CFG:
        skip("body_fakes", "no body / export")
        return
    for x, sw in E.COPIES.items():
        arms = plan(sw)
        for k in [k_ for k_ in E.ARM_KEYS if sw[k_]]:
            a = copy.deepcopy(arms[k])
            o, calls, log = body_run(a, sw, measured=k != "B")
            want = [f"{w[0]}@{w[1]}{w[2]}" for w in a["windows_fixed"]]
            smoke = a.get("e34_smoke")
            check(f"fake_body {x} {k}: its windows exactly ({len(want)}), warmup {'off' if smoke else 'on'}",
                  windows_called(log) == want and calls.count("warmup") == (0 if smoke else 1) and "status" not in o, (windows_called(log), want, o.get("status")))
            check(f"fake_body {x} {k}: the capacity record once, the smoke without the ladder",
                  calls.count("e34_capacity_rec") == 1 and calls.count("smoke") == 1 and o["smoke"]["ladder_on"] is False)
            n_or = 0 if smoke else (2 if k == "B" else 1)
            check(f"fake_body {x} {k}: oracle x{n_or}, long oracle x1, prefix x{0 if smoke else 1}, TF x{2 if (k == 'B' or smoke) else 1}",
                  calls.count("oracle") == n_or and calls.count("long_oracle") == 1 and calls.count("prefix_test") == (0 if smoke else 1)
                  and calls.count("tforced") == (2 if (k == "B" or smoke) else 1), calls)
            check(f"fake_body {x} {k}: prefill probe only on B", calls.count("prefill_probe") == (1 if k == "B" else 0))
            check(f"fake_body {x} {k}: no PROF profile, no sibling probe", "profile_window" not in calls and "sibling_probe" not in calls)
            if smoke:
                check(f"fake_body {x} {k}: the smoke TF stores its own reference (tf_ref + tf_aa)", "tf_ref" in o and "tf_aa" in o and "tf_vs_q" not in o)
            elif k != "B":
                check(f"fake_body {x} {k}: TF vs B's reference + the gate recorded", "tf_vs_q" in o and "tf_gate" in o and "tf_ref" not in o)
    o, calls, log = body_run(copy.deepcopy(ARMS["S16B2"]), E.COPIES["full"], d1f="E34 read-back: block 4096 != 2048")
    check("fake_body: an E34 read-back real failure -> NOT_INTEGRATED before the smoke",
          o.get("status") == "NOT_INTEGRATED" and o.get("not_integrated") and "smoke" not in calls, (o.get("status"), calls[:8]))
    o, calls, log = body_run(copy.deepcopy(ARMS["S16"]), E.COPIES["full"], smoke_gate="FAIL_SMOKE_GARBAGE")
    check("fake_body: a smoke FAIL stops the arm before any window (NO_GO_SMOKE)", o.get("status") == "NO_GO_SMOKE" and not windows_called(log))
    if FX is not None and "d1_readback_failure" in FNS:
        ns = {"D1": D1, "E34": E, "FAKE": False}
        exec(FNS["d1_readback_failure"], ns)
        log_b = "\n".join(FX["log_lines"])
        for k, wk, lg, want in (("B", FX["workers"], log_b, None), ("S16", FX["workers"], log_b, "E34 read-back"),
                                ("S16B2", _s16b2_workers(FX["workers"]), log_b.replace("(4096, 4096, 4096, 8, 4096, 1024)", "(2048, 2048, 2048, 8, 2048, 512)"), None)):
            o = {"workers": wk, "dflash_readback": {"checks": dict(FX["dflash_checks"])}, "config_readback": {"log": {}}, "env_readback_raw": None}
            r_ = ns["d1_readback_failure"](o, ARMS[k], lg)
            check(f"d1_readback_failure {k} on the GPU-derived record: {want or 'no failure'}",
                  (r_ is None) if want is None else (isinstance(r_, str) and r_.startswith(want)), (r_, (o.get("e34_readback") or {}).get("why")))


class _BSrv:
    def __init__(self, log):
        self._log, self.ready_sec, self.argv = log, 1.0, []

    def log(self):
        return self._log


def boot_ns(outcomes, arm, left_sec=10 * 3600):
    tries, seen = list(outcomes), []
    ns = {"N": N, "T": T, "D1": D1, "re": re, "K100_CFG": K100CFG, "Q_I8": {"on": False, "want": False, "dropped": None},
          "NB4T_BUILD_ENV": {}, "NB1_EXTRA": {}, "NB1_ENV": {}, "NB1_KV": {}, "ARM_OF": {}, "CUR": {}, "BOOT": {}, "RESP": {},
          "NBDF_READY_SEC": 1200, "left": lambda: left_sec, "log": lambda *a, **k: None, "tree_check": lambda a: {"ok": True},
          "boot_gate": lambda: {"ok": True}, "shm_names": lambda: [], "shm_cleanup": lambda x: {}, "q_readback": lambda s, a: {},
          "OOM_ERR": re.compile(r"CUDA out of memory|OutOfMemoryError"), "XID": re.compile(r"Xid \d+"),
          "gated": lambda o, need=(), extra=(), expected_rows=4: dict(o, gated_need=need, gated_rows=expected_rows),
          "dflash_rb": lambda a, t: {}, "arm_watched": lambda b, c, n: b}

    class _Mon:
        def __init__(self, n):
            self.last, self.stop_ev = None, type("E", (), {"set": lambda s: None})()

        def start(self):
            pass
    ns["LoadMonitor"] = _Mon

    def mfs(name, kv, port, kvb):
        seen.append({"name": name, "port": port, "kvb": kvb, "env": dict(ns["NB1_ENV"].get(name) or {}), "extra": list(ns["NB1_EXTRA"].get(name) or []),
                     "expect": dict(ns["ARM_OF"][name]["expect"])})
        return _BSrv(tries[min(len(seen) - 1, len(tries) - 1)][1])
    ns["merlin_fn_server"] = mfs
    ns["with_servers"] = lambda srvs, ready, b: {"status": tries[min(len(seen) - 1, len(tries) - 1)][0]}
    for fn_ in ("_k100_retry", "arm_boot"):
        exec(FNS[fn_], ns)
    o = ns["arm_boot"](arm, lambda ss: {})
    return o, seen, ns


def t_boot_fakes():
    if not FNS or not K100CFG:
        skip("boot_fakes", "no body / export")
        return
    o, seen, ns = boot_ns([("SERVER_DIED", "torch.OutOfMemoryError: CUDA out of memory (TF prompt logprobs)"), ("DONE", "ok")], copy.deepcopy(ARMS["S16B2"]))
    check("boot S16B2: first try at 7,933,526,016 with SW 512 + EMBED_HOST + its SSM / block argv",
          seen[0]["kvb"] == PIN and seen[0]["env"].get("DFLASH_DRAFT_SW_BLOCK") == "512" and seen[0]["env"].get("DFLASH_DRAFT_EMBED_HOST") == "1"
          and seen[0]["extra"][-4:] == ["--mamba-ssm-cache-dtype", "bfloat16", "--block-size", "2048"], seen[0])
    check("boot S16B2: an OOM at any stage -> one rung lower (7,665,090,560), port + 1, the plan kept",
          [s_["kvb"] for s_ in seen] == [PIN, 7665090560] and seen[1]["port"] == seen[0]["port"] + 1
          and seen[1]["env"].get("DFLASH_DRAFT_SW_BLOCK") == "512" and seen[1]["expect"].get("DFLASH_DRAFT_SW_BLOCK") == "512"
          and seen[1]["extra"] == seen[0]["extra"], seen)
    check("boot S16B2: the oracle in the plan -> gated vs B's rows", o.get("gated_need") == ("q",) and o.get("gated_rows") == 4, o)
    o, seen, ns = boot_ns([("LAUNCH_FAILED", "CUDA out of memory")] * 6, copy.deepcopy(ARMS["S16B2W"]))
    check("boot S16B2W: the D ladder ends after its 4 rungs", [s_["kvb"] for s_ in seen] == [PIN, 7665090560, 7396655104, 6859784192])
    o, seen, ns = boot_ns([("SERVER_DIED", "CUDA out of memory\nNVRM: Xid 79"), ("DONE", "ok")], copy.deepcopy(ARMS["S16"]))
    check("boot: an OOM with an Xid -> no retry", len(seen) == 1)
    o, seen, ns = boot_ns([("NOT_INTEGRATED", "E34 read-back")], copy.deepcopy(ARMS["S16"]))
    check("boot: a non-OOM failure never retries", len(seen) == 1 and o.get("status") == "NOT_INTEGRATED")
    o, seen, ns = boot_ns([("DONE", "ok")], copy.deepcopy(plan(E.COPIES["smoke"])["S16B2"]))
    check("boot SMOKE S16B2: no oracle -> the hard-only evidence", o.get("gated_need") == () and o.get("gated_rows") is None)


class _Seq:
    def __init__(self, apply_ok=True):
        self.state, self.ring, self.log, self.error, self.apply_rec, self.revert_rec = "applied", True, [], None, {}, {}
        self.verify_rec, self.apply_ok = {}, apply_ok

    def apply(self):
        return (True, "applied") if self.apply_ok else (False, "NOT_INTEGRATED: seam apply failed")

    def check(self, k):
        return True, "FULLY_PATCHED"

    def before_arm(self, a):
        return True, "FULLY_PATCHED"

    def finish(self):
        self.state = "reverted"


def main_ns(switch=None, gate=True, outs=None):
    calls, R_ = [], {"sections": {"A0": {"merge": {"ok": True}}}, "setup": {}}
    arms_ = plan(switch)
    E34R_ = {}

    def runner(k):
        calls.append((k, arms_[k]["streams"]))
        return dict((outs or {}).get(k) or {"status": "DONE", "pin": PIN})

    def step(sec, name, est, fn):
        r_ = dict(fn() or {})
        if r_.get("status") in (None, "NOT_RUN"):
            r_["status"] = "DONE"
        R_["sections"].setdefault(sec, {})[name] = r_
        return r_
    ns = {"N": N, "P": P, "T": T, "D1": D1, "E34": E, "R": R_, "ARMS": arms_, "ISO": {}, "NBDF": {}, "NBD1": {}, "FAKE": True, "step": step,
          "save": lambda: None, "log": lambda *a, **k: None, "nbd1_finalize": lambda: None, "left": lambda: 10 * 3600,
          "merlin_install": lambda: {"kv": {"MERLIN_TREE": "/x"}}, "apply_60490": lambda: {"status": "DONE", "ok": True},
          "k100_setup": lambda: {"ok": True}, "sha_gate": lambda t: (True, "ok"), "SEAM_API": object(), "SEAM_LOAD": {},
          "_tree_py": lambda: (_ for _ in ()).throw(RuntimeError("no tree")), "DRAFTER": {"ok": True, "status": None, "why": "x"},
          "KV_KIT_GATE": {"passed": gate, "per_k": {}, "why": None if gate else ["closed"]}, "PSEQ": None, "SSEQ": None,
          "run_arm": runner, "NBD1_SWITCH": E.switch_map(switch), "E34R": E34R_, "NBDF_ARMS": {"D3s": D3S}, "T_START": time.time(),
          "json": json, "time": time}
    for fn_ in ("_arm", "skip_arm", "_patch_rec", "_seam_rec", "_arm_rec", "nbd1_main", "e34_w_replan"):
        exec(FNS[fn_], ns)
    buf = io.StringIO()
    with redirect_stdout(buf):
        ns["nbd1_main"](runner=runner, pseq=_Seq(), sseq=_Seq())
    return calls, R_, ns, buf.getvalue()


def t_main_fakes():
    if not FNS or "e34_w_replan" not in FNS:
        skip("main_fakes", "no body")
        return
    cap = {"pool": 1080, "mc_131072": 1080 / 81.0, "attn_block": 2048}
    c_, R_, ns_, out = main_ns(switch=E.COPIES["full"], outs={"S16B2": {"status": "DONE", "e34_capacity": dict(cap, pool=1240, mc_131072=1240 / 81.0)}})
    w_exp = (1240 - 2) // (20 + 17)
    check("main FULL: B > S16 > S16B2 > S16B2W, S16B2W re-planned from S16B2's boot (W = %d)" % w_exp,
          [k for k, _ in c_] == list(E.ARM_KEYS) and c_[-1] == ("S16B2W", w_exp) and ns_["E34R"]["w"]["replanned"] is True
          and ns_["ARMS"]["S16B2W"]["captures"] == [4 * i for i in range(1, w_exp + 1)] and "[LIVE" in out, (c_, ns_["E34R"].get("w")))
    c_, R_, ns_, out = main_ns(switch=E.COPIES["full"], outs={"S16B2": {"status": "DONE", "e34_capacity": cap}})
    check("main FULL: W from S16B2 = the planned 29 -> no re-plan", c_[-1] == ("S16B2W", 29) and "replanned" not in ns_["E34R"]["w"], ns_["E34R"].get("w"))
    c_, R_, ns_, out = main_ns(switch=E.COPIES["full"], outs={"S16B2": {"status": "NOT_INTEGRATED"}})
    check("main FULL: S16B2 unreadable -> the CPU planner W, recorded, S16B2W still runs", c_[-1] == ("S16B2W", 29)
          and "CPU planner" in ns_["E34R"]["w"]["source"], ns_["E34R"].get("w"))
    c_, R_, ns_, out = main_ns(switch=E.COPIES["smoke"])
    check("main SMOKE: only S16B2 boots, the others SKIPPED_SWITCH_OFF", c_ == [("S16B2", 24)]
          and all(next(iter(R_["sections"][k].values())).get("status") == "SKIPPED_SWITCH_OFF" for k in ("B", "S16", "S16B2W")), c_)
    c_, R_, ns_, out = main_ns(switch=E.COPIES["full"], gate=False)
    check("main: the closed gate -> no arm boots", c_ == [] and all(next(iter(R_["sections"][k].values())).get("status") == "SKIPPED_KV_KIT_GATE" for k in E.ARM_KEYS))
    c_, R_, ns_, out = main_ns(switch=E.COPIES["full"], outs={"B": {"status": "NO_GO_SMOKE"}})
    check("main: a failed B never skips the levers", [k for k, _ in c_] == list(E.ARM_KEYS))


def t_window_fakes():
    if not FNS:
        skip("window_fakes", "no body")
        return

    def hw(port, s, band_, n_, arm):
        return {"status": "STEADY", "tok_s": 700.0, "running_max": float(n_), "prefix_hit_rate": 0.02}
    ns = {"N": Mod("N", [], {"kv_bound": lambda r: False, "capacity_window_status": lambda *a: None, "LONG_BANDS": ("90k", "116k")}),
          "D1": D1, "E34": E, "CUR": {}, "traceback": traceback, "win_need": lambda b_, n_: 10.0, "due": lambda s: True, "boot_left": lambda: 999,
          "hist_window": hw, "_watched": _W, "_live": lambda o, k_, v: o.__setitem__(k_, v), "d1_window_ledger": lambda o, r, b_, n_: {},
          "_arm_rec": lambda k_: {"steady": {"40k@16": {"tok_s": 650.0}, "40k@20W": {"tok_s": 680.0}}}, "T_START": time.time() - 600, "time": time}
    exec(FNS["d1_live"], ns)
    exec(FNS["run_window"], ns)
    o = {"steady": {}, "items_skipped": []}
    s = type("S", (), {"port": 1, "alive": lambda self: True})()
    buf = io.StringIO()
    with redirect_stdout(buf):
        ns["run_window"](o, s, dict(ARMS["S16B2W"]), "40k", 22, "W")
    ln = [l for l in buf.getvalue().splitlines() if l.startswith("[LIVE")]
    check("window: one [LIVE] line with the prefix hit and B's W window", len(ln) == 1 and "prefix hit 0.02" in ln[0] and "vs B window 40k@20W" in ln[0], ln)
    if "e34_item_live" in FNS and "_live" in FNS:
        ns2 = {"E34": E, "time": time, "T_START": time.time(), "LIVE": {}, "save": lambda: None}
        exec(FNS["e34_item_live"], ns2)
        exec(FNS["_live"], ns2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            ns2["_live"]({"arm": "S16"}, "oracle", {"vs_q": {"status": "EXACT", "rows": []}})
            ns2["_live"]({"arm": "S16"}, "census", {"x": 1})
        lines = [l for l in buf.getvalue().splitlines() if l.startswith("[LIVE")]
        check("_live: a [LIVE] line after a test item, none for a non-test key", len(lines) == 1 and "S16 oracle" in lines[0], lines)


# ------------------------------------------------------------------ 6. facts / export / copy block / notebook
def t_facts():
    if not FACTS.is_file():
        skip("facts", f"no facts {FACTS}")
        return
    f = json.loads(FACTS.read_text(encoding="utf-8"))
    check("facts_lib_sha", f["e34_lib_sha256"] == hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest())
    check("facts_nbd1_lib_sha", f["nbd1_lib_sha256"] == hashlib.sha256(Path(D1.__file__).read_bytes()).hexdigest() == "af0f795f685fe63f6169e5171f25f5d35f25bf66cfc6538c1272554d3190bad3")
    check("facts_base_reproduced", f["nbd1_reproduced_sha256"] == "abbf21dfa9cda439de7e557e5782526512f9087c7e36a0c3f70625c30d13bb6d")
    check("facts_size", all(c["sha256"] for c in f["copies"].values()) and f["bytes"] < 400_000)
    check("facts_copies", set(f["copies"]) == {"full", "smoke"} and all(f["copies"][x]["switch"] == E.COPIES[x] for x in f["copies"]))
    check("facts_budgets", f["budgets_min"]["full"] == E.budget_for(plan(E.COPIES["full"]), E.COPIES["full"])
          and f["budgets_min"]["smoke"] == E.budget_for(plan(E.COPIES["smoke"]), E.COPIES["smoke"]))
    check("facts_sources", set(f["dataset_sources"]) == {"foysalemonshanto/q27-e25-nb0-tf-ref", "foysalemonshanto/q27-e34-payload", "foysalemonshanto/q27-e34-tf-long"}
          and len(f["model_sources"]) == 4 and any("dflash2v2" in m for m in f["model_sources"]))
    check("facts_payload", f["payload_dataset"]["uploaded"] is False and f["payload_dataset"]["files"]["e34_lib.py"] == f["e34_lib_sha256"]
          and f["tf_long_dataset"]["uploaded"] is False and set(f["tf_long_dataset"]["traces"]) == {k for k, _ in E.TF_LONG}
          and all(60000 <= v["len"] < 131072 for v in f["tf_long_dataset"]["traces"].values()))
    if BTXT:
        check("facts_body_sha", f["body_file_sha256"] == hashlib.sha256(BTXT.encode("utf-8")).hexdigest())
        m = re.search(r'E34_CFG = json.loads\(r"""(.*?)"""\)', BTXT)
        check("body cfg: the TF dataset sha = the facts", m is not None and json.loads(m.group(1))["tf_long_sha256"] == f["tf_long_dataset"]["sha256"])
    if not EXTXT:
        skip("export", f"no export {EXPORT}")
        return
    check("facts_export_sha", f["export_sha256"] == hashlib.sha256(EXTXT.encode("utf-8")).hexdigest())
    eb = EXTXT.encode("utf-8")
    cells, pos = [], 0
    for nb_ in f["cell_bytes"]:
        cells.append(eb[pos:pos + nb_].decode("utf-8"))
        pos += nb_ + 2
    check("export_cells_split", pos - 2 == len(eb) and len(cells) == f["cells"] - 1)
    c1 = cells[1]
    blk = f["copies"]["full"]["copy_block"]
    check("copy block at the top of CELL 1 (E34_ON + E34_BUDGET_MIN)", c1.startswith(blk) and EXTXT.count("E34_ON = ") == 1
          and EXTXT.count("E34_BUDGET_MIN = ") == 1 and "NBD1_ON =" not in EXTXT)
    ns = {}
    exec("\n".join(l for l in blk.splitlines() if not l.startswith("#")), ns)
    check("copy block runs: E34_ON = the FULL switch, the budget", ns["E34_ON"] == E.COPIES["full"] and ns["E34_BUDGET_MIN"] == f["budgets_min"]["full"])
    check("cell0_e18r2", hashlib.sha256(cells[0].encode()).hexdigest() == f["e18_r2_cell0_sha256"])
    _cm = re.search(r"^COMP = next\(.*?\n.*?\n", c1, re.M | re.S)
    check("export COMP: one statement, no bare competitions-only COMP", _cm is not None and EXTXT.count("\nCOMP = ") == 1
          and 'COMP = INP / "competitions" / "arc-prize-2026-arc-agi-2"\n' not in EXTXT)
    if _cm:
        import tempfile
        for lay in ("competitions/arc-prize-2026-arc-agi-2", "arc-prize-2026-arc-agi-2", None):
            with tempfile.TemporaryDirectory() as td:
                inp = Path(td)
                if lay:
                    (inp / lay).mkdir(parents=True)
                    (inp / lay / "arc-agi_evaluation_challenges.json").write_text("{}")
                ns = {"INP": inp}
                exec(_cm.group(0), ns)
                want = inp / (lay or "competitions/arc-prize-2026-arc-agi-2")
                check(f"export COMP layout {lay or 'neither (default)'}", ns["COMP"] == want and (lay is None or (ns["COMP"] / "arc-agi_evaluation_challenges.json").is_file()), ns["COMP"])
    check("export budget / deadman from the copy block", 'or E34_BUDGET_MIN * 60)' in c1 and "NBDF_DEADMAN_SEC = (E34_BUDGET_MIN + 5) * 60" in c1)
    check("export stub payload loader", '_e34_ds("e34_cell7_body.py", NBDF_BODY_SHA)' in EXTXT and "q27-e34-payload" in EXTXT and "_nbd1_ds = _e34_ds" in EXTXT)
    if NBF.is_file():
        nb = json.loads(NBF.read_text(encoding="utf-8"))
        src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        check("notebook_equals_export", "\n\n".join(src) == EXTXT and hashlib.sha256(NBF.read_bytes()).hexdigest() == f["sha256"])
        p = NBF.with_name(f["copies"]["smoke"]["notebook"])
        if p.is_file():
            sx = ["".join(cc["source"]) for cc in json.loads(p.read_text(encoding="utf-8"))["cells"] if cc["cell_type"] == "code"]
            check("smoke copy: only the copy block differs", sx[0] == src[0] and sx[2:] == src[2:]
                  and sx[1] == src[1].replace(blk, f["copies"]["smoke"]["copy_block"]) and hashlib.sha256(p.read_bytes()).hexdigest() == f["copies"]["smoke"]["sha256"])
    else:
        skip("notebook", f"no notebook {NBF} (the units kit ships the export)")


def main():
    for t in (t_plan, t_launch, t_capacity_readback, t_misc, t_body_static, t_body_fakes, t_boot_fakes, t_main_fakes, t_window_fakes, t_facts):
        try:
            t()
        except Exception as e:
            check(t.__name__ + "_crashed", False, f"{type(e).__name__}: {e} | {traceback.format_exc()[-1500:]}")
    print(f"{RES['pass']}/{RES['total']} PASS", flush=True)
    sys.exit(0 if RES["pass"] == RES["total"] else 1)


if __name__ == "__main__":
    main()
