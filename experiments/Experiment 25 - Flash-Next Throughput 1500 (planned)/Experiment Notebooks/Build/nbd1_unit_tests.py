"""E25 NB-D1 unit tests (stdlib only; local Windows Python and the Py3.12 Kaggle CPU units kit).

  python nbd1_unit_tests.py [--libs DIR [DIR ...]] [--merlin-src DIR] [--body FILE] [--notebook FILE] [--facts FILE] [--export FILE]
                            [--fixture FILE]

Covers: the arm plan (B = NBDF-K100 D3s + EMBED_HOST + QSA 128 / 64; every lever = B + exactly one change; mutations refused),
nbdf_lib.launch_check per arm on the launch env (Merlin's build() env + NB1_ENV after it), the switches and the three copies,
the read-backs (on the NB-S1 GPU fixture: QSA / GDN env in 4 workers, EMBED_HOST, IDX, LP, SPC, seqs, K, DYN), the D ladder
with EMBED_HOST first (retry_plan + arm_boot with fakes: an OOM at any stage steps down, an Xid does not, the expect keeps the
QSA keys), the QSA env, the time plan and budget, the [LIVE] line, the NB-D1 kit gate, arm_body with fakes (light items per
arm), nbd1_main with fakes (copies, B off, the closed gate, no cross-arm skip), run_window with fakes (step down on any band
for a lever), the body / facts / export / switch line (read from the code export). The facts / export / body checks need no
notebook (the units kit ships the export, not the .ipynb). Prints `FAIL <name>: <why>`, `SKIP <name>: <why>` and
`<passed>/<total> PASS` last.
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


LIBS = _arg("--libs", [str(HERE), str(E27B)], many=True)
for d in reversed(LIBS):
    sys.path.insert(0, d)
sys.dont_write_bytecode = True
MERLIN = Path(_arg("--merlin-src", r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src"))
BODY = Path(_arg("--body", str(HERE / "nbd1_cell7_body.py")))
NBF = Path(_arg("--notebook", str(HERE.parent / "Q27-XHIGH-E25_NBD1.ipynb")))
FACTS = Path(_arg("--facts", str(HERE / "nbd1_build_facts.json")))
EXPORT = Path(_arg("--export", str(HERE / "nbd1_code_export.py")))
FIXTURE = Path(_arg("--fixture", str(HERE / "nbd1_log_fixture.json")))
NBDF_LIB_B2 = "b014775162fb55401826fa8c498f7009b5d47ad15cc5cb365ab23e5ed37b1929"
PIN = 7933526016

import nb4t_lib as T  # noqa: E402
import nbdf_lib as N  # noqa: E402
import nbd1_lib as D1  # noqa: E402
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
        print(f"FAIL {name}: {why}"[:700], flush=True)


def skip(name, why):
    print(f"SKIP {name}: {why}", flush=True)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


# ------------------------------------------------------------------ the NBDF-K100 D3s (nbdf_lib, configured as the K100 body)
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
ARMS = D1.arm_plan(N, D3S, seam_env_k4=dict(SEAM), switch=None)
FX = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.is_file() else None


def plan(sw=None):
    return D1.arm_plan(N, D3S, seam_env_k4=dict(SEAM), switch=sw)


def launch_env(a, nb1_env=True):
    """The launch env as the notebook's merlin_fn_server builds it: Merlin build()'s plan env (partition + the QSA workspace
    at 256 / 256, launch.py:419-420) and then env.update(NB1_ENV[name]) = the arm's env."""
    env = {"VLLM_PP_LAYER_PARTITION": "25,23", "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "256", "VLLM_E2_QSA_LOGITS_GRANULE_MB": "256",
           "KV_BYTES": str(a["kv_bytes"])}
    if nb1_env:
        env.update(a["env"])
    return env


def launch_argv(a):
    return list(a["extra"]) + ["--kv-cache-memory-bytes", str(a["kv_bytes"])]


# ------------------------------------------------------------------ 1. the arm plan
def t_plan():
    check("plan_order", tuple(ARMS) == D1.ARM_KEYS, list(ARMS))
    check("plan_valid", D1.check_arm_plan(N, T, ARMS, D3S) is True)
    b = ARMS["B"]
    check("B_is_D3s_argv_spec_captures_pin", all(b[f] == D3S[f] for f in ("extra", "spec", "captures", "k", "method", "kv_bytes", "q_label")))
    ed = {x: b["env"].get(x) for x in set(b["env"]) | set(D3S["env"]) if b["env"].get(x) != D3S["env"].get(x)}
    check("B_env_EH_QSA_only", ed == {"DFLASH_DRAFT_EMBED_HOST": "1", "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "128",
                                      "VLLM_E2_QSA_LOGITS_GRANULE_MB": "64"}, ed)
    check("B_no_indexer_fp8_no_gdn", "--attention-config" not in b["extra"] and not any(k.startswith("VLLM_GDN") for k in b["env"]))
    check("B_no_draft_head", not b.get("draft_head") and "DRAFT_HEAD" not in b["env"] and b["spec"].get("draft_lm_head_quant") is None)
    check("B_dflash_k3_bf16_flash_attn", b["spec"]["method"] == "dflash" and b["spec"]["num_speculative_tokens"] == 3
          and b["spec"]["kv_cache_dtype"] == "auto" and b["spec"]["attention_backend"] == "FLASH_ATTN")
    check("B_captures_4_96_seqs_24", b["captures"] == [4 * i for i in range(1, 25)] and T.last_value(b["extra"], "--max-num-seqs") == "24"
          and b["streams"] == 24, b["captures"])
    check("B_eh_first_flag", b["d1_eh_first"] is True and all(a["d1_eh_first"] for a in ARMS.values()))
    check("B_expect_has_qsa_eh", b["expect"]["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] == "128" and b["expect"]["VLLM_E2_QSA_LOGITS_GRANULE_MB"] == "64"
          and b["expect"]["DFLASH_DRAFT_EMBED_HOST"] == "1" and b["expect"]["VLLM_GDN_DECODE_FASTPATH"] is None)
    q = ARMS["Q256"]
    check("Q256_env_256", {k: q["env"][k] for k in D1.QSA_KEYS} == D1.QSA_MERLIN and q["extra"] == b["extra"] and q["d1_change"]["kind"] == "env")
    for k in ("G1", "G2"):
        a = ARMS[k]
        check(f"{k}_env", a["env"].get(D1.GDN_ENV[k]) == "1" and a["extra"] == b["extra"]
              and {x for x in a["env"] if a["env"].get(x) != b["env"].get(x)} == {D1.GDN_ENV[k]})
    d4 = ARMS["D4s"]
    check("D4s_k4_captures_5_120", d4["k"] == 4 and d4["spec"]["num_speculative_tokens"] == 4 and d4["captures"] == [5 * i for i in range(1, 25)]
          and N.captures_from_argv(d4["extra"]) == d4["captures"] and T.spec_from_argv(d4["extra"]) == d4["spec"])
    check("D4s_spec_otherwise_B", dict(d4["spec"], num_speculative_tokens=3) == b["spec"] and d4["env"] == b["env"])
    dy = ARMS["DYN"]
    check("DYN_schedule_captures", dy["spec"]["num_speculative_tokens_per_batch_size"] == [[1, 8, 4], [9, 24, 3]] and dy["k"] == 4
          and dy["captures"] == sorted({5 * x for x in range(1, 9)} | {4 * x for x in range(9, 25)}) and dy["coverage"]["covers_all"], dy["captures"])
    check("DYN_dense", D1.dense_schedule(D1.DYN_SCHEDULE, 24, 4)[1:] == [4] * 8 + [3] * 16)
    check("IDX_argv", ARMS["IDX"]["extra"] == b["extra"] + ["--attention-config", '{"indexer_kv_dtype":"fp8"}']
          and not any(str(x).startswith("--attention-config.") for x in ARMS["IDX"]["extra"]))
    check("LP_argv", ARMS["LP"]["extra"] == b["extra"] + ["--max-num-batched-tokens", "2048", "--long-prefill-token-threshold", "1024",
                                                         "--long-prefill-token-threshold-adaptive"])
    check("SPC_argv", ARMS["SPC"]["extra"] == b["extra"] + ["--enable-mamba-shared-prefix-checkpoint"])
    for k, s in (("S28", 28), ("S32", 32)):
        a = ARMS[k]
        check(f"{k}_seqs_captures", T.last_value(a["extra"], "--max-num-seqs") == str(s) and a["streams"] == s
              and a["captures"] == [4 * x for x in range(1, s + 1)] and N.captures_from_argv(a["extra"]) == a["captures"]
              and T.spec_from_argv(a["extra"]) == b["spec"] and a["expect_cfg"]["max_num_seqs"] == s)
        check(f"{k}_windows", a["windows_fixed"] == [["20k", s, ""], ["90k", 8, ""], ["116k", 6, ""]], a["windows_fixed"])
    for k in ("Q256", "D4s", "DYN", "G1", "G2", "IDX", "LP", "SPC"):
        check(f"{k}_windows_light", ARMS[k]["windows_fixed"] == [["8k", 16, "#1"], ["20k", 24, ""], ["90k", 8, ""], ["116k", 6, ""]],
              ARMS[k]["windows_fixed"])
        check(f"{k}_no_oracle_tf_aa", not any(D1.item_on(ARMS[k], x) for x in ("oracle", "tf", "long_oracle", "prefix", "ladder", "prefill_probe")))
    check("B_items_light", D1.item_on(b, "prefill_probe") and not any(D1.item_on(b, x) for x in ("oracle", "tf", "long_oracle", "prefix", "ladder")))
    check("sibling_only_SPC", [k for k, a in ARMS.items() if D1.item_on(a, "sibling")] == ["SPC"])
    check("ports_distinct", len({a["port"] for a in ARMS.values()}) == len(ARMS) and max(a["port"] for a in ARMS.values()) + 8 <= 8355)
    check("pins_equal", {a["kv_bytes"] for a in ARMS.values()} == {PIN})
    check("every_arm_eh_qsa", all(a["env"].get("DFLASH_DRAFT_EMBED_HOST") == "1" for a in ARMS.values())
          and all({k: a["env"][k] for k in D1.QSA_KEYS} == (D1.QSA_MERLIN if kk == "Q256" else D1.QSA_B) for kk, a in ARMS.items()))
    check("every_window_set_90k_116k", all(any(w[0] == "90k" for w in a["windows_fixed"]) and any(w[0] == "116k" for w in a["windows_fixed"])
                                           for a in ARMS.values()))
    check("base_must_be_d3s", raises(lambda: D1.arm_plan(N, ARMS0["Q"]), ValueError))
    for nm, mut in (("extra_flag", lambda a: a["IDX"]["extra"].append("--foo")),
                    ("env_change", lambda a: a["LP"]["env"].update(NCCL_ALGO="Ring")),
                    ("two_changes", lambda a: a["G1"]["env"].update(VLLM_GDN_SHARED_ALIGN_INDEX="1")),
                    ("gdn_on_B", lambda a: a["B"]["env"].update(VLLM_GDN_DECODE_FASTPATH="1")),
                    ("qsa_off_B", lambda a: a["B"]["env"].pop("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB")),
                    ("eh_off_lever", lambda a: a["G2"]["env"].pop("DFLASH_DRAFT_EMBED_HOST")),
                    ("seqs_wrong", lambda a: a["S28"]["extra"].__setitem__(-1, "30")),
                    ("never_do", lambda a: a["SPC"]["extra"].append("--no-enable-prefix-caching")),
                    ("pin", lambda a: a["IDX"].update(kv_bytes=7665090560)),
                    ("k4_captures", lambda a: a["D4s"].update(captures=[4 * i for i in range(1, 25)])),
                    ("lever_windows_no_116k", lambda a: a["G1"].update(windows_fixed=[["8k", 16, "#1"], ["90k", 8, ""]])),
                    ("lever_tf", lambda a: a["LP"]["d1_items"].update(tf=True)),
                    ("draft_head", lambda a: a["Q256"]["env"].update(DRAFT_HEAD="int8")),
                    ("order", lambda a: a.update({"X": a.pop("B")}))):
        a2 = copy.deepcopy(ARMS)
        mut(a2)
        check(f"plan_refuses_{nm}", raises(lambda: D1.check_arm_plan(N, T, a2, D3S), ValueError))


# ------------------------------------------------------------------ 2. launch_check per arm (the final launch env / argv)
def t_launch():
    for k, a in ARMS.items():
        bad = N.launch_check(a, launch_env(a), launch_argv(a), a["kv_bytes"])
        check(f"launch_check_{k}", bad == [], bad)
    b = ARMS["B"]
    bad = N.launch_check(b, launch_env(b, nb1_env=False) | {x: b["env"][x] for x in b["env"] if x not in D1.QSA_KEYS}, launch_argv(b), PIN)
    check("launch_check_catches_merlin_256_on_B", any("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" in x for x in bad) and
          any("VLLM_E2_QSA_LOGITS_GRANULE_MB" in x for x in bad), bad)
    e = launch_env(b)
    e["VLLM_GDN_DECODE_FASTPATH"] = "1"
    check("launch_check_catches_gdn_on_B", any("VLLM_GDN_DECODE_FASTPATH" in x for x in N.launch_check(b, e, launch_argv(b), PIN)))
    e = launch_env(b)
    e.pop("DFLASH_DRAFT_EMBED_HOST")
    check("launch_check_catches_no_eh", any("DFLASH_DRAFT_EMBED_HOST" in x for x in N.launch_check(b, e, launch_argv(b), PIN)))
    a = ARMS["S32"]
    x = launch_argv(a)
    x[x.index("--max-num-seqs", len(x) - 6) + 1] = "24"
    check("launch_check_catches_seqs", any("max-num-seqs" in y for y in N.launch_check(a, launch_env(a), x, PIN)))
    a = ARMS["D4s"]
    check("launch_check_catches_spec", bool(N.launch_check(a, launch_env(a), launch_argv(ARMS["B"]), PIN)))


# ------------------------------------------------------------------ 3. switches, copies, time, budget
def t_switch_time():
    check("switch_absent_all_on", D1.switch_map(None) == {k: True for k in D1.SWITCH_KEYS})
    check("switch_invalid_all_on", D1.switch_map("x") == {k: True for k in D1.SWITCH_KEYS} and D1.switch_record("x")["invalid_not_a_dict"])
    check("switch_missing_key_off", D1.switch_map({"B": True}) == dict({k: False for k in D1.SWITCH_KEYS}, B=True))
    check("switch_record_unknown", D1.switch_record({"B": True, "ZZ": 1})["unknown_keys"] == ["ZZ"])
    want = {"a": ["B", "BAA", "PROF", "Q256", "D4s", "DYN"], "b": ["B", "G1", "G2", "IDX", "LP"], "c": ["B", "S28", "S32", "SPC"]}
    for x, sw in D1.COPIES.items():
        check(f"copy_{x}_keys", [k for k in D1.SWITCH_KEYS if sw[k]] == want[x] and set(sw) == set(D1.SWITCH_KEYS), sw)
    check("copies_cover_every_arm_once", sorted(k for x in "abc" for k in want[x] if k in D1.LEVERS) == sorted(D1.LEVERS))
    check("owners", D1.OWNERS == {"a": "foysalemonshanto", "b": "hosen42", "c": "foysalemonshanto"})
    runtimes, needs = {}, {}
    for x, sw in D1.COPIES.items():
        a = plan(sw)
        check(f"copy_{x}_plan_valid", D1.check_arm_plan(N, T, a, D3S) is True)
        w = D1.copy_walk(a, sw)
        runtimes[x], needs[x] = w["runtime_min"], D1.copy_need_min(a, sw)
        check(f"copy_{x}_all_run", w["all_run"] and w["runs"] == [k for k in D1.ARM_KEYS if sw[k]], w)
        check(f"copy_{x}_runtime_le_150", w["runtime_min"] <= D1.COPY_RUNTIME_MAX, w["runtime_min"])
        check(f"copy_{x}_runtime_ge_100", w["runtime_min"] >= 100, w["runtime_min"])
        for k, arm in a.items():
            check(f"copy_{x}_{k}_cap_holds_body", arm["cap_min"] >= arm["time_model"]["body"] + 1, (arm["cap_min"], arm["time_model"]["body"]))
    top = max(max(runtimes.values()), max(needs.values())) + D1.START_RESERVE_MIN
    check("budget_formula", D1.BUDGET_MIN == 5 * -(-top // 5) and D1.DEADMAN_MIN == D1.BUDGET_MIN + 5, (D1.BUDGET_MIN, runtimes, needs))
    a = plan(D1.COPIES["a"])
    check("copyA_B_full_windows_prof", a["B"]["windows_fixed"] == D1.B_WINDOWS_FULL and a["B"]["prof_windows"] == [["8k", 16], ["20k", 24]])
    a = plan(D1.COPIES["b"])
    check("copyB_B_light_no_prof", a["B"]["windows_fixed"] == D1.LEVER_WINDOWS and a["B"]["prof_windows"] == [])
    check("prof_plan_only_B", D1.prof_plan({"key": "B"}, D1.COPIES["a"]) == [("8k", 16), ("20k", 24)]
          and D1.prof_plan({"key": "G1"}, D1.COPIES["a"]) == [] and D1.prof_plan({"key": "B"}, D1.COPIES["c"]) == [])
    rows = plan(D1.COPIES["a"])["B"]["time_model"]["items"]
    check("time_model_prof_need", any(r[0] == "profile_20k@24" and r[2] is not None for r in rows), rows)
    w = D1.copy_walk(plan(D1.SWITCH_DEFAULT), D1.SWITCH_DEFAULT)
    check("all_on_kit_notebook_walk_records_skips", w["runs"][0] == "B" and len(w["walk"]) == len(D1.ARM_KEYS), w)
    w = D1.copy_walk(plan(D1.COPIES["b"]), D1.COPIES["b"], budget_min=60)
    check("walk_skips_when_short", w["skipped"] and not w["all_run"], w)


# ------------------------------------------------------------------ 4. read-backs (the NB-S1 GPU fixture)
def _env(env_raw, **over):
    e = copy.deepcopy(env_raw)
    for w in e["results"]:
        for k, v in over.items():
            if v is None:
                w["env"].pop(k, None)
            else:
                w["env"][k] = v
    return e


def t_readback():
    if FX is None:
        skip("readback_fixture", f"no fixture {FIXTURE}")
        return
    ll = FX["log_lines"]
    log = "\n".join([ll["nondefault"], ll["init"], ll["granule"]])
    crb = {"log": FX["config_readback_log"]}
    drb = {"checks": dict(FX["dflash_checks"])}
    env_b = _env(FX["env_readback_raw"], VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="128", VLLM_E2_QSA_LOGITS_GRANULE_MB="64")
    check("fx_4_workers", len(D1.env_rows(env_b)) == 4 and {r for r, _ in D1.env_rows(env_b)} == {0, 1, 2, 3})
    r = D1.readback(ARMS["B"], log, env_b, crb, {}, drb)
    check("rb_B_ok_on_gpu_fixture", r["ok"] and not r["real_failure"], r.get("why"))
    check("rb_B_logged_1024_24", r["logged_max_num_batched_tokens"] == 1024 and r["logged_max_num_seqs"] == 24 and r["nondefault_args_line"])
    check("rb_B_qsa_verified", r["qsa"]["label"] == "QSA_VERIFIED" and r["qsa"]["granule_logged_mib"] == [256], r["qsa"])
    r = D1.readback(ARMS["B"], log, FX["env_readback_raw"], crb, {}, drb)
    check("rb_B_qsa_256_recorded_base_mismatch", r["ok"] and any("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" in x for x in r["base_mismatch"])
          and r["qsa"]["label"] == "QSA_ENV_MISMATCH", r)
    r = D1.readback(ARMS["G1"], log, _env(FX["env_readback_raw"], VLLM_GDN_DECODE_FASTPATH="1"), crb, {}, drb)
    check("rb_lever_qsa_256_recorded_own_change_ok", r["ok"] and r["base_mismatch"] and r["qsa"]["label"] == "QSA_ENV_MISMATCH", r)
    env_q = _env(FX["env_readback_raw"], VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="256", VLLM_E2_QSA_LOGITS_GRANULE_MB="256")
    check("rb_Q256_ok_256", D1.readback(ARMS["Q256"], log, env_q, crb, {}, drb)["ok"])
    check("rb_Q256_128_fails", D1.readback(ARMS["Q256"], log, env_b, crb, {}, drb)["real_failure"])
    e3 = copy.deepcopy(env_b)
    e3["results"] = e3["results"][:3]
    r = D1.readback(ARMS["B"], log, e3, crb, {}, drb)
    check("rb_3_workers_unreadable_recorded", r["ok"] and set(D1.D1_ENV_KEYS) <= set(r["unreadable"]) and r["qsa"]["label"] == "UNVERIFIED_QSA", r)
    check("rb_rpc_error_recorded", D1.readback(ARMS["B"], log, {"error": "x"}, crb, {}, drb)["ok"])
    for k in ("G1", "G2"):
        key = D1.GDN_ENV[k]
        check(f"rb_{k}_ok", D1.readback(ARMS[k], log, _env(env_b, **{key: "1"}), crb, {}, drb)["ok"])
        check(f"rb_{k}_missing_fails", D1.readback(ARMS[k], log, env_b, crb, {}, drb)["real_failure"])
        check(f"rb_{k}_on_B_fails", D1.readback(ARMS["B"], log, _env(env_b, **{key: "1"}), crb, {}, drb)["real_failure"])
    r = D1.readback(ARMS["B"], log, env_b, crb, {}, {"checks": dict(drb["checks"], embed_host_as_planned=False)})
    check("rb_eh_false_recorded_base_mismatch", r["ok"] and any("EMBED_HOST" in x for x in r["base_mismatch"]), r)
    r = D1.readback(ARMS["B"], log, env_b, crb, {}, {})
    check("rb_eh_missing_recorded", r["ok"] and "embed_host_as_planned" in r["unreadable"], r)
    wk = lambda dt: {"ranks": [{"rank": i, "kv_cache_specs": [{"example": "model.layers.3.self_attn.indexer.compressed_key_cache", "dtype": dt}]}
                               for i in range(4)]}
    check("rb_IDX_fp8_ok", D1.readback(ARMS["IDX"], log, env_b, crb, wk("torch.float8_e4m3fn"), drb)["ok"])
    check("rb_IDX_bf16_fails", D1.readback(ARMS["IDX"], log, env_b, crb, wk("torch.bfloat16"), drb)["real_failure"])
    check("rb_IDX_none_nondefault_not_fp8_fails", D1.readback(ARMS["IDX"], log, env_b, crb, {}, drb)["real_failure"])
    check("rb_IDX_fp8_on_B_fails", D1.readback(ARMS["B"], log, env_b, crb, wk("torch.float8_e4m3fn"), drb)["real_failure"])
    lp_log = log.replace("'max_num_batched_tokens': 1024", "'max_num_batched_tokens': 2048, 'long_prefill_token_threshold': 1024, "
                                                            "'long_prefill_token_threshold_adaptive': True")
    check("rb_fixture_has_mbt", "'max_num_batched_tokens': 1024" in log)
    check("rb_LP_ok", D1.readback(ARMS["LP"], lp_log, env_b, crb, {}, drb)["ok"])
    check("rb_LP_on_B_log_fails", D1.readback(ARMS["LP"], log, env_b, crb, {}, drb)["real_failure"])
    check("rb_LP_log_on_B_fails", D1.readback(ARMS["B"], lp_log, env_b, crb, {}, drb)["real_failure"])
    spc_log = log.replace("'enable_prefix_caching': True,", "'enable_prefix_caching': True, 'enable_mamba_shared_prefix_checkpoint': True,")
    check("rb_SPC_ok", D1.readback(ARMS["SPC"], spc_log, env_b, crb, {}, drb)["ok"])
    check("rb_SPC_missing_fails", D1.readback(ARMS["SPC"], log, env_b, crb, {}, drb)["real_failure"])
    check("rb_SPC_on_B_fails", D1.readback(ARMS["B"], spc_log, env_b, crb, {}, drb)["real_failure"])
    r = D1.readback(ARMS["SPC"], ll["granule"], env_b, crb, {}, drb)
    check("rb_no_nondefault_line_recorded", r["ok"] and any("non-default" in u for u in r["unreadable"]), r)
    crb32 = {"log": dict(FX["config_readback_log"], max_num_seqs=32)}
    check("rb_S32_ok", D1.readback(ARMS["S32"], log, env_b, crb32, {}, drb)["ok"])
    check("rb_S32_24_fails", D1.readback(ARMS["S32"], log, env_b, crb, {}, drb)["real_failure"])
    sp4 = dict(FX["config_readback_log"]["speculative_config"], num_speculative_tokens=4)
    crb4 = {"log": dict(FX["config_readback_log"], speculative_config=sp4)}
    check("rb_D4s_ok", D1.readback(ARMS["D4s"], log, env_b, crb4, {}, drb)["ok"])
    check("rb_D4s_k3_fails", D1.readback(ARMS["D4s"], log, env_b, crb, {}, drb)["real_failure"])
    crbd = {"log": dict(FX["config_readback_log"], speculative_config=dict(sp4, num_speculative_tokens_per_batch_size=[[1, 8, 4], [9, 24, 3]]))}
    check("rb_DYN_ok", D1.readback(ARMS["DYN"], log, env_b, crbd, {}, drb)["ok"])
    check("rb_DYN_no_schedule_fails", D1.readback(ARMS["DYN"], log, env_b, crb4, {}, drb)["real_failure"])
    check("rb_DYN_override_line_fails", D1.readback(ARMS["DYN"], log + "\nWARNING Dynamic speculative decoding ... Overriding cudagraph_mode to PIECEWISE",
                                                    env_b, crbd, {}, drb)["real_failure"])
    check("rb_schedule_on_D4s_fails", D1.readback(ARMS["D4s"], log, env_b, crbd, {}, drb)["real_failure"])
    check("granule_parse", D1.granule_logged(ll["granule"]) == [256])
    check("fixture_windows_b_ref", all(abs(FX["windows"][k]["tok_s"] - v) < 0.05 for k, v in D1.B_REF["tok_s"].items()), FX["windows"])
    sib = D1.sibling_label({"prefix_tokens": 6900, "hits": {"C": 6900}}, 4096)
    check("sibling_label", sib["label"] == "SPC_JUNCTION_HIT" and D1.sibling_label({"prefix_tokens": 6900, "hits": {"C": 4096}}, 4096)["label"] == "BLOCK_BOUNDARY_ONLY")
    m = D1.parse_metrics('vllm:kv_cache_usage_perc{engine="0"} 0.5\nvllm:num_preemptions_total 2\nvllm:prompt_tokens_by_source_total{source="computed"} 10\n')
    check("parse_metrics", m["vllm:kv_cache_usage_perc"] == 0.5 and m["vllm:num_preemptions_total"] == 2 and m["by_source"]["computed"] == 10)
    ls = D1.ledger_summary([(0.0, m), (1.0, dict(m, **{"vllm:num_preemptions_total": 3.0}))], pool=540)
    check("ledger_summary", ls["preemptions"] == 1.0 and ls["blocks_in_use_max"] == 270, ls)


# ------------------------------------------------------------------ 5. retry plan, live line, gate, merlin
def t_misc():
    dr = [[7933526016, {"DFLASH_DRAFT_EMBED_HOST": "1"}], [7665090560, {"DFLASH_DRAFT_EMBED_HOST": "1"}],
          [7396655104, {"DFLASH_DRAFT_EMBED_HOST": "1"}], [6859784192, {"DFLASH_DRAFT_EMBED_HOST": "1"}]]
    check("retry_plan_eh_first_skips_same_pin", [x[0] for x in D1.retry_plan(dr, PIN, True)] == [7665090560, 7396655104, 6859784192])
    check("retry_plan_without_eh_first_unchanged", D1.retry_plan(dr, PIN, False) == dr)
    w = {"status": "STEADY", "tok_s": 650.0, "mean_acceptance_length": 3.2, "running_max": 16.0, "preemptions": 0.0, "level": 16}
    ln = D1.live_line(37.25, "B", "8k@16#1", w)
    check("live_line_B_ref", ln == "[LIVE 37.2] B 8k@16#1 tok/s 650.0 | AL 3.20 | running 16 | preempt 0 | vs B (ref) +3.0 %", ln)
    brec = {"steady": {"20k@24": {"tok_s": 800.0}}}
    ln = D1.live_line(90.0, "S28", "20k@28", dict(w, tok_s=840.0, running_max=28.0), brec)
    check("live_line_vs_B_same_band", ln.endswith("| vs B +5.0 %") and "running 28" in ln, ln)
    ln = D1.live_line(91.0, "S32", "20k@32", {"status": "SKIPPED_BOOT_BUDGET"}, brec)
    check("live_line_skipped", "tok/s None" in ln and "[SKIPPED_BOOT_BUDGET]" in ln and "vs B n/a" in ln, ln)
    check("live_line_regex", re.match(r"^\[LIVE [\d.]+\] \S+ \S+ tok/s \S+ \| AL \S+ \| running \S+ \| preempt \S+ \| vs B ", D1.live_line(1, "G1", "90k@8", w, {})))
    base = {"passed": True, "per_k": {"D3s": True}, "kv_summary": {}}
    rep = {"notebook_sha256": "abc", "verdict": {"ALL_OK": True, "checks": {k: True for k in D1.KIT_KEYS}}}
    check("gate_ok", D1.gate_from_report(base, rep, "abc")["passed"] is True)
    check("gate_stale_sha", D1.gate_from_report(base, rep, "zzz")["passed"] is False)
    check("gate_missing_key", D1.gate_from_report(base, dict(rep, verdict={"ALL_OK": True, "checks": {}}), "abc")["passed"] is False)
    check("gate_not_all_ok", D1.gate_from_report(base, dict(rep, verdict=dict(rep["verdict"], ALL_OK=False)), "abc")["passed"] is False)
    check("gate_base_closed", D1.gate_from_report(dict(base, passed=False), rep, "abc")["passed"] is False)
    cg = D1.closed_gate(base)
    check("closed_gate", cg["passed"] is False and cg["per_k"] == {"D3s": True} and base["passed"] is True)
    u = MERLIN / "vllm/v1/spec_decode/dynamic/utils.py"
    if u.is_file():
        ns = {}
        try:
            src = u.read_text(encoding="utf-8")
            tree = ast.parse(src)
            fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_dynamic_sd_schedule_lookup"]
            imp = "\n".join(ast.get_source_segment(src, n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
            helpers = "\n\n".join(ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef))
            exec(compile("from __future__ import annotations\n" + "\n".join(l for l in imp.splitlines() if "vllm" not in l) + "\n" + helpers,
                         str(u), "exec"), ns)
            got = ns["build_dynamic_sd_schedule_lookup"](D1.DYN_SCHEDULE, 24, 4) if fn else None
            check("dense_schedule_equals_merlin", got is not None and list(got) == D1.dense_schedule(D1.DYN_SCHEDULE, 24, 4), got)
        except Exception as e:
            skip("dense_schedule_equals_merlin", f"merlin utils not executable standalone: {type(e).__name__}: {e}"[:200])
    else:
        skip("dense_schedule_equals_merlin", f"no {u}")


# ------------------------------------------------------------------ 6. the body: static checks + fakes
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
    check("body_compiles", compile(b, "nbd1_cell7_body.py", "exec") is not None)
    for marker in ('oom = bool(o["oom_in_log"]) and not o["xid_in_log"]', 'D1.retry_plan(K100_CFG["d_retry"], int(K100_CFG["q_pin"]), bool(arm.get("d1_eh_first")))',
                   'RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P," + ",".join(D1.D1_ENV_KEYS)',
                   'arm["expect"] = dict(arm["expect"], **{w: arm["env"].get(w, N.WATCHED_DEFAULT.get(w)) for w in N.WATCHED})',
                   'for c in (N.CONC_LADDER if D1.item_on(arm, "ladder") else ()):', 'if D1.item_on(arm, "oracle"):',
                   'd1_live(o, arm, key, r)                       # E25 NB-D1: [LIVE', 'D1.prof_plan(arm, NBD1_SWITCH)', 'secs=float(D1.PROF_SECS)',
                   'D1.check_arm_plan(N, T, ARMS, NBDF_ARMS["D3s"])', 'nbd1_main()', 'globals().get("NBD1_ON")',
                   '(band in ("40k",) + N.LONG_BANDS or arm.get("lever"))', 'return gated(o, need=(), expected_rows=None)',
                   '_d1f = d1_readback_failure(o, arm, t)', 'NBD1_LIB_SRC = _nbd1_ds("nbd1_lib.py", NBD1_LIB_SHA)'):
        check(f"body_marker {marker[:60]}", b.count(marker) == 1, b.count(marker))
    check("body_old_oom_rule_gone", 'oom = o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]' not in b)
    check("body_no_bare_ladder", "    for c in N.CONC_LADDER:" not in b)
    check("body_nbdf_main_not_called", "\n    nbdf_main()" not in b and b.count("nbd1_main()") == 1)
    m = re.search(r'^KV_KIT_GATE = json.loads\(r"""(.*?)"""\)', b, re.M)
    g = json.loads(m.group(1)) if m else {}
    fx = json.loads(FACTS.read_text(encoding="utf-8")) if FACTS.is_file() else {}
    check("body_kv_gate_matches_facts", m is not None and g.get("passed") == fx.get("gpu_ready") and "nbd1_kit_checks" in g, g.get("why"))
    if EXTXT:
        i_b = EXTXT.find('eng["ML"].build("flash-next"')
        i_e = EXTXT.find("env.update(NB1_ENV.get(name) or {})              # E25 NB1: after build()")
        check("export_nb1_env_after_merlin_build", 0 < i_b < i_e, (i_b, i_e))


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
             "spec_decoding_lines": lambda *a: [], "kv_bound": lambda *a: False}
    ns = {"N": Mod("N", log, nspec), "T": Mod("T", log, {}), "P": Mod("P", log, {"parse_log": lambda *a: []}),
          "L": Mod("L", log, {}), "D1": D1, "json": json, "re": re, "time": time, "traceback": traceback,
          "FAKE": False, "CUR": {"kv_bytes": PIN}, "LIVE": {}, "KV_KIT_GATE": {}, "Q_AA": {}, "Q_ACC": {}, "Q_TF_AA": {},
          "PREFILL": {"tok_s": 4000.0, "measured": measured}, "K100_CFG": K100CFG, "NBD1_SWITCH": D1.switch_map(switch),
          "RESP": {"counts": {}, "arm": None}, "MEM_RX": re.compile(r"Model loading took"), "_watched": _W, "ROWS": {}}

    def _live(o, k, v):
        o[k] = v
    for nm, ret in (("due", True), ("short_gate_failed", False), ("_mem_gate", (True, "ok")), ("quality_gate", "PASS"),
                    ("band_level", lambda o, b: (99, {"source": "fake"})), ("boot_facts", lambda s, o, n: o),
                    ("jit_kernels", lambda *a: []), ("smoke", lambda p, a: {"acceptance": {}, "ladder_on": D1.item_on(a, "ladder")}),
                    ("run_window", lambda o, s, a, b, n, tag="": f"{b}@{n}{tag}"), ("profile_window", {"status": "DONE"}),
                    ("i8_readback", {"label": "I8_ENGAGED"}), ("k100_boot_gates", {"real_failure": False}),
                    ("win_need", 10.0), ("d1_readback_failure", d1f), ("d1_records", {"rec": True}),
                    ("sibling_probe", {"label": {"label": "SPC_JUNCTION_HIT"}}),
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
    never = ("oracle", "long_oracle", "prefix_test", "tforced", "admission_probes")
    for x, sw in D1.COPIES.items():
        arms = plan(sw)
        for k in [k_ for k_ in D1.ARM_KEYS if sw[k_]]:
            a = copy.deepcopy(arms[k])
            o, calls, log = body_run(a, sw, measured=k != "B")
            want = [f"{w[0]}@{w[1]}{w[2]}" for w in a["windows_fixed"]]
            prof = [(c[1][2], c[1][3], c[2].get("secs")) for c in log if c[0] == "profile_window"]
            check(f"fake_body copy {x} {k}: smoke without the ladder, warmup, its windows, no oracle / TF / A/A items",
                  calls.count("smoke") == 1 and o["smoke"]["ladder_on"] is False and calls.count("warmup") == 1 and windows_called(log) == want
                  and not any(c in calls for c in never) and o.get("quality_gate") == "NOT_IN_PLAN_NB_D1" and o.get("tf_plan") == "NOT_IN_PLAN_NB_D1"
                  and o.get("long_gate") == "NOT_IN_PLAN_NB_D1" and o.get("prefix_gate") == "NOT_IN_PLAN_NB_D1" and "status" not in o,
                  (calls, windows_called(log), want, o.get("status")))
            check(f"fake_body copy {x} {k}: prefill probe only on B", calls.count("prefill_probe") == (1 if k == "B" else 0), calls.count("prefill_probe"))
            check(f"fake_body copy {x} {k}: PROF {'30-s 8k@16 / 20k@24' if (k == 'B' and sw['PROF']) else 'none'}",
                  prof == ([("8k", 16, 30.0), ("20k", 24, 30.0)] if (k == "B" and sw["PROF"]) else []), prof)
            check(f"fake_body copy {x} {k}: sibling probe only on SPC", calls.count("sibling_probe") == (1 if k == "SPC" else 0))
            check(f"fake_body copy {x} {k}: the NB-D1 read-back ran, the records written",
                  calls.count("d1_readback_failure") == 1 and o.get("d1_records") == {"rec": True})
            check(f"fake_body copy {x} {k}: two long-band resets only", calls.count("reset_prefix") == 2, calls.count("reset_prefix"))
    o, calls, log = body_run(copy.deepcopy(ARMS["G1"]), D1.COPIES["b"], measured=False)
    check("fake_body a lever runs the prefill probe when B's rate was never measured (B off)", calls.count("prefill_probe") == 1)
    full = copy.deepcopy(ARMS["IDX"])
    full["d1_items"] = {}
    o, calls, log = body_run(full, D1.COPIES["b"])
    check("fake_body harness contrast: a plan with every item on calls the oracle, the long oracle, the prefix test and TF",
          calls.count("oracle") >= 1 and calls.count("long_oracle") == 1 and calls.count("prefix_test") == 1 and calls.count("tforced") >= 1, calls)
    for k in ("B", "D4s"):
        o, calls, log = body_run(copy.deepcopy(ARMS[k]), D1.SWITCH_DEFAULT, smoke_gate="FAIL_SMOKE_GARBAGE")
        check(f"fake_body {k}: a smoke FAIL (hard label) stops the arm before any window (NO_GO_SMOKE)",
              o.get("status") == "NO_GO_SMOKE" and not windows_called(log), (o.get("status"), windows_called(log)))
        o, calls, log = body_run(copy.deepcopy(ARMS[k]), D1.SWITCH_DEFAULT, d1f="NB-D1 read-back: QSA mismatch")
        check(f"fake_body {k}: an NB-D1 read-back real failure -> NOT_INTEGRATED before the smoke",
              o.get("status") == "NOT_INTEGRATED" and o.get("not_integrated") and "smoke" not in calls, (o.get("status"), calls[:8]))
    if FX is not None:                       # the body's d1_readback_failure on the GPU fixture (real D1.readback)
        ns = {"D1": D1, "FAKE": False}
        exec(FNS["d1_readback_failure"], ns)
        ll = FX["log_lines"]
        log_ = chr(10).join([ll["nondefault"], ll["init"], ll["granule"]])
        base_o = {"config_readback": {"log": FX["config_readback_log"]}, "workers": {}, "dflash_readback": {"checks": dict(FX["dflash_checks"])}}
        o = dict(copy.deepcopy(base_o), env_readback_raw=FX["env_readback_raw"])
        r_ = ns["d1_readback_failure"](o, ARMS["B"], log_)
        check("d1_readback_failure B with Merlin 256 / 256 in the workers: recorded (recorded_gates), no stop",
              r_ is None and any("NB-D1 base settings" in g for g in o.get("recorded_gates") or []), (r_, o.get("recorded_gates")))
        o = dict(copy.deepcopy(base_o), env_readback_raw=_env(FX["env_readback_raw"], VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="128", VLLM_E2_QSA_LOGITS_GRANULE_MB="64"))
        r_ = ns["d1_readback_failure"](o, ARMS["Q256"], log_)
        check("d1_readback_failure Q256 with 128 / 64 in the workers: a real failure (its own change)", isinstance(r_, str) and "NB-D1 read-back" in r_, r_)
        o = dict(copy.deepcopy(base_o), env_readback_raw=_env(FX["env_readback_raw"], VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="128", VLLM_E2_QSA_LOGITS_GRANULE_MB="64"))
        check("d1_readback_failure B on the production env: no failure, nothing recorded", ns["d1_readback_failure"](o, ARMS["B"], log_) is None
              and not o.get("recorded_gates") and o["d1_readback"]["qsa"]["label"] == "QSA_VERIFIED", o.get("d1_readback"))
    o, calls, log = body_run(copy.deepcopy(ARMS["Q256"]), D1.SWITCH_DEFAULT, smoke_gate="INCONCLUSIVE_SMOKE_SHORT")
    check("fake_body a non-FAIL smoke label is recorded, the arm measures on", "status" not in o and windows_called(log), o.get("recorded_gates"))


# ---- arm_boot with fakes: the D ladder (EMBED_HOST first, an OOM at any stage)
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
        seen.append({"name": name, "port": port, "kvb": kvb, "env": dict(ns["NB1_ENV"].get(name) or {}),
                     "expect": dict(ns["ARM_OF"][name]["expect"]), "build_env": dict(ns["NB4T_BUILD_ENV"].get(name) or {})})
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
    o, seen, ns = boot_ns([("SERVER_DIED", "torch.OutOfMemoryError: CUDA out of memory (first request after ready)"), ("DONE", "ok")],
                          copy.deepcopy(ARMS["B"]))
    check("boot B: the first try at 7,933,526,016 with EMBED_HOST + QSA 128/64 in NB1_ENV, no DRAFT_HEAD",
          seen[0]["kvb"] == PIN and seen[0]["env"].get("DFLASH_DRAFT_EMBED_HOST") == "1" and seen[0]["env"].get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB") == "128"
          and seen[0]["env"].get("VLLM_E2_QSA_LOGITS_GRANULE_MB") == "64" and seen[0]["build_env"] == {}, seen[0])
    check("boot B: an OOM after ready (SERVER_DIED) -> the next try one rung lower (7,665,090,560; not the same-pin EH rung), port + 1",
          [s_["kvb"] for s_ in seen] == [PIN, 7665090560] and seen[1]["port"] == seen[0]["port"] + 1 and len(o["tries"]) == 2, seen)
    check("boot B: the retry keeps the QSA / GDN keys in the expect (launch_check + env read-back)",
          seen[1]["expect"].get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB") == "128" and "VLLM_GDN_DECODE_FASTPATH" in seen[1]["expect"]
          and seen[1]["expect"].get("DFLASH_DRAFT_EMBED_HOST") == "1", seen[1]["expect"])
    check("boot B: no oracle in the plan -> gated on the hard-only evidence", o.get("gated_need") == () and o.get("gated_rows") is None, o)
    o, seen, ns = boot_ns([("SERVER_DIED", "CUDA out of memory\nNVRM: Xid 79"), ("DONE", "ok")], copy.deepcopy(ARMS["G1"]))
    check("boot: OOM with an Xid (device fault) -> no retry", len(seen) == 1, seen)
    o, seen, ns = boot_ns([("LAUNCH_FAILED", "CUDA out of memory")] * 6, copy.deepcopy(ARMS["S32"]))
    check("boot: the D ladder ends after its last rung (4 tries: 7,933,526,016 / 7,665,090,560 / 7,396,655,104 / 6,859,784,192)",
          [s_["kvb"] for s_ in seen] == [PIN, 7665090560, 7396655104, 6859784192], [s_["kvb"] for s_ in seen])
    o, seen, ns = boot_ns([("NO_GO_SMOKE", "FAIL_SMOKE_GARBAGE")], copy.deepcopy(ARMS["D4s"]))
    check("boot: a non-OOM failure never retries", len(seen) == 1 and o.get("status") == "NO_GO_SMOKE", (seen, o.get("status")))
    o, seen, ns = boot_ns([("SERVER_DIED", "CUDA out of memory"), ("DONE", "ok")], copy.deepcopy(ARMS["Q256"]))
    check("boot Q256: the retry keeps 256 / 256", seen[1]["env"].get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB") == "256" and len(seen) == 2, seen)


# ---- nbd1_main with fakes
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


def main_ns(switch=None, gate=True, outs=None, seam_ok=True, drafter_ok=True):
    calls, R_ = [], {"sections": {"A0": {"merge": {"ok": True}}}, "setup": {}}
    arms_ = plan(switch)

    def runner(k):
        calls.append(k)
        return dict((outs or {}).get(k) or {"status": "DONE", "pin": PIN, "smoke_gate": "PASS"})

    def step(sec, name, est, fn):
        r_ = dict(fn() or {})
        if r_.get("status") in (None, "NOT_RUN"):
            r_["status"] = "DONE"
        R_["sections"].setdefault(sec, {})[name] = r_
        return r_
    ns = {"N": N, "P": P, "T": T, "D1": D1, "R": R_, "ARMS": arms_, "ISO": {}, "NBDF": {}, "NBD1": {}, "FAKE": True, "step": step,
          "save": lambda: None, "log": lambda *a, **k: None, "nbd1_finalize": lambda: None, "left": lambda: 10 * 3600,
          "merlin_install": lambda: {"kv": {"MERLIN_TREE": "/x"}}, "apply_60490": lambda: {"status": "DONE", "ok": True},
          "k100_setup": lambda: {"ok": True}, "sha_gate": lambda t: (True, "ok"), "SEAM_API": object(), "SEAM_LOAD": {},
          "_tree_py": lambda: (_ for _ in ()).throw(RuntimeError("no tree")), "DRAFTER": {"ok": drafter_ok, "status": None, "why": "x"},
          "KV_KIT_GATE": {"passed": gate, "per_k": {}, "why": None if gate else ["closed"]}, "PSEQ": None, "SSEQ": None,
          "run_arm": runner, "NBD1_SWITCH": D1.switch_map(switch)}
    for fn_ in ("_arm", "skip_arm", "_patch_rec", "_seam_rec", "_arm_rec", "nbd1_main"):
        exec(FNS[fn_], ns)
    ns["nbd1_main"](runner=runner, pseq=_Seq(), sseq=_Seq(seam_ok))
    return calls, R_, ns


def status_of(R_, k):
    return next(iter(R_["sections"].get(k, {"": {}}).values())).get("status")


def t_main_fakes():
    if not FNS:
        skip("main_fakes", "no body")
        return
    c_, R_, ns_ = main_ns()
    check("main all on: every arm in order", c_ == list(D1.ARM_KEYS), c_)
    for x, sw in D1.COPIES.items():
        c_, R_, ns_ = main_ns(switch=sw)
        want = [k for k in D1.ARM_KEYS if sw[k]]
        check(f"main copy {x}: exactly its arms launch, the others SKIPPED_SWITCH_OFF", c_ == want
              and all(status_of(R_, k) == "SKIPPED_SWITCH_OFF" for k in D1.ARM_KEYS if not sw[k]), (c_, want))
        pr = (R_["sections"].get("PROF") or {}).get("nbd1_prof") or {}
        check(f"main copy {x}: PROF recorded ({'from B' if sw['PROF'] else 'SKIPPED_SWITCH_OFF'})",
              pr.get("status") == ("NOT_RUN" if sw["PROF"] else "SKIPPED_SWITCH_OFF"), pr)
    c_, R_, ns_ = main_ns(switch=dict(D1.COPIES["c"], B=False))
    check("main B off: the levers still launch", c_ == ["S28", "S32", "SPC"] and status_of(R_, "B") == "SKIPPED_SWITCH_OFF", c_)
    c_, R_, ns_ = main_ns(switch=D1.COPIES["b"], outs={"B": {"status": "NO_GO_SMOKE"}, "G1": {"status": "NOT_INTEGRATED", "not_integrated": True}})
    check("main: a failed arm never skips the next (no cross-arm skip)", c_ == ["B", "G1", "G2", "IDX", "LP"], c_)
    c_, R_, ns_ = main_ns(gate=False)
    check("main: the closed gate -> no arm boots, every arm SKIPPED_KV_KIT_GATE", c_ == [] and all(status_of(R_, k) == "SKIPPED_KV_KIT_GATE" for k in D1.ARM_KEYS), c_)
    c_, R_, ns_ = main_ns(seam_ok=False)
    check("main: the seam failing -> every arm NOT_INTEGRATED (all are DFlash2)", c_ == [] and status_of(R_, "B") == "NOT_INTEGRATED", c_)
    c_, R_, ns_ = main_ns(switch={})
    check("main: every arm off -> no setup", c_ == [] and ns_["NBDF"].get("stop", "").startswith("ALL_ARMS_SWITCHED_OFF"), ns_["NBDF"])


# ---- run_window with fakes: a lever's KV-bound window steps down on any band; the [LIVE] line
def t_window_fakes():
    if not FNS:
        skip("window_fakes", "no body")
        return
    for k, band, n in (("S32", "20k", 32), ("G1", "8k", 16), ("B", "8k", 16)):
        calls = []
        first = {"v": True}

        def hw(port, s, band_, n_, arm):
            calls.append(n_)
            kb = first["v"]
            first["v"] = False
            return {"status": "INCONCLUSIVE_KV_BOUND" if kb else "STEADY", "tok_s": 700.0, "kv_bound": kb, "running_max": float(n_)}
        ns = {"N": Mod("N", [], {"kv_bound": lambda r: bool(r.get("kv_bound")), "capacity_window_status": lambda *a: None,
                                 "LONG_BANDS": ("90k", "116k")}),
              "D1": D1, "CUR": {}, "traceback": traceback, "win_need": lambda b_, n_: 10.0, "due": lambda s: True, "boot_left": lambda: 999,
              "hist_window": hw, "_watched": _W, "_live": lambda o, k_, v: o.__setitem__(k_, v), "d1_window_ledger": lambda o, r, b_, n_: {},
              "_arm_rec": lambda k_: {"steady": {"8k@16#1": {"tok_s": 650.0}, "20k@24": {"tok_s": 760.0}}}, "T_START": time.time() - 600,
              "time": time}
        exec(FNS["d1_live"], ns)
        exec(FNS["run_window"], ns)
        o = {"steady": {}, "items_skipped": []}
        s = type("S", (), {"port": 1, "alive": lambda self: True})()
        buf = io.StringIO()
        with redirect_stdout(buf):
            ns["run_window"](o, s, dict(ARMS[k]), band, n, "")
        lines = [l for l in buf.getvalue().splitlines() if l.startswith("[LIVE")]
        if k == "B":
            check("window B 8k: no step down at 8k (base rule: 40k / long bands only)", calls == [16], calls)
        else:
            check(f"window {k} {band}: KV-bound -> one step down ({n} -> {n - 1}), never skipped", calls == [n, n - 1], calls)
        check(f"window {k}: a [LIVE] line per measured window", len(lines) == len(calls) and all(" tok/s 700.0 " in l and "| vs B " in l for l in lines), lines)
    ns["due"] = lambda s: False
    exec(FNS["run_window"], ns)
    o = {"steady": {}, "items_skipped": []}
    buf = io.StringIO()
    with redirect_stdout(buf):
        ns["run_window"](o, s, dict(ARMS["G2"]), "116k", 6, "")
    check("window skipped (budget): recorded + a [LIVE] line", o["steady"]["116k@6"]["status"] == "SKIPPED_BOOT_BUDGET"
          and "[SKIPPED_BOOT_BUDGET]" in buf.getvalue(), buf.getvalue())


# ------------------------------------------------------------------ 7. facts / export / switch line / notebook
def t_facts():
    if not FACTS.is_file():
        skip("facts", f"no facts {FACTS}")
        return
    f = json.loads(FACTS.read_text(encoding="utf-8"))
    check("facts_lib_sha", f["nbd1_lib_sha256"] == hashlib.sha256((Path(D1.__file__).read_bytes())).hexdigest())
    check("facts_nbdf_lib_b2", f["nbdf_lib_sha256"] == NBDF_LIB_B2)
    check("facts_seam_b1", f["seam_sha256"] == "341d859be5b2cc619247ff7b645304b9473d09f07995358ead2ee02be116c60b")
    check("facts_size", f["bytes"] < 400_000 and all(c["sha256"] for c in f["copies"].values()), f["bytes"])
    check("facts_budget", f["budget_min"] == D1.BUDGET_MIN and f["deadman_min"] == D1.DEADMAN_MIN)
    check("facts_copies", set(f["copies"]) == {"a", "b", "c"} and all(f["copies"][x]["switch_line"] == "NBD1_ON = " + repr(D1.COPIES[x])
                                                                    and f["copies"][x]["owner"] == D1.OWNERS[x] for x in f["copies"]))
    check("facts_copy_sources", all(set(c["dataset_sources"]) == {"foysalemonshanto/q27-e25-nb0-tf-ref", "foysalemonshanto/q27-e25-nbd1-payload"}
                                    and len(c["model_sources"]) == 4 and any("dflash2v2" in m for m in c["model_sources"])
                                    and any("klee100" in m for m in c["model_sources"]) for c in f["copies"].values()), f["copies"]["b"]["dataset_sources"])
    check("facts_payload", f["payload_dataset"]["slug"] == "foysalemonshanto/q27-e25-nbd1-payload" and f["payload_dataset"]["uploaded"] is False
          and f["payload_dataset"]["files"]["nbd1_lib.py"] == f["nbd1_lib_sha256"])
    if BTXT:
        check("facts_body_sha", f["body_file_sha256"] == hashlib.sha256(BTXT.encode("utf-8")).hexdigest())
    if not EXTXT:
        skip("export", f"no export {EXPORT}")
        return
    check("facts_export_sha", f["export_sha256"] == hashlib.sha256(EXTXT.encode("utf-8")).hexdigest())
    eb = EXTXT.encode("utf-8")
    cells, pos = [], 0
    for nb_ in f["cell_bytes"]:
        cells.append(eb[pos:pos + nb_].decode("utf-8"))
        pos += nb_ + 2
    check("export_cells_split", pos - 2 == len(eb) and len(cells) == f["cells"] - 1, (pos, len(eb)))
    c1 = cells[1]
    line = "NBD1_ON = " + repr(D1.SWITCH_DEFAULT)
    check("switch_line_once_top_cell1", c1.splitlines()[4] == line and EXTXT.count("NBD1_ON = ") == 1, c1.splitlines()[:6])
    lit = ast.literal_eval(line.split(" = ", 1)[1])
    check("switch_line_literal_default", lit == D1.SWITCH_DEFAULT)
    ns = {}
    exec(line, ns)
    check("switch_line_runs", ns["NBD1_ON"] == D1.SWITCH_DEFAULT)
    check("switch_line_not_in_body_or_lib", (not BTXT or "NBD1_ON =" not in BTXT) and "NBD1_ON =" not in Path(D1.__file__).read_text(encoding="utf-8"))
    check("cell0_e18r2", hashlib.sha256(cells[0].encode()).hexdigest() == f["e18_r2_cell0_sha256"])
    check("export_budget_deadman", f"or {D1.BUDGET_MIN} * 60)" in c1 and f"NBDF_DEADMAN_SEC = {D1.DEADMAN_MIN * 60}" in c1)
    check("export_stub_payload_loader", '_nbd1_ds("nbd1_cell7_body.py", NBDF_BODY_SHA)' in EXTXT and "q27-e25-nbd1-payload" in EXTXT)
    if NBF.is_file():
        nb = json.loads(NBF.read_text(encoding="utf-8"))
        src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        check("notebook_equals_export", "\n\n".join(src) == EXTXT and hashlib.sha256(NBF.read_bytes()).hexdigest() == f["sha256"])
        for x, c in f["copies"].items():
            p = NBF.with_name(c["notebook"])
            if p.is_file():
                nbx = json.loads(p.read_text(encoding="utf-8"))
                sx = ["".join(cc["source"]) for cc in nbx["cells"] if cc["cell_type"] == "code"]
                check(f"copy_{x}_only_switch_line_differs", sx[0] == src[0] and sx[2:] == src[2:]
                      and sx[1] == src[1].replace(line, c["switch_line"]) and hashlib.sha256(p.read_bytes()).hexdigest() == c["sha256"])
    else:
        skip("notebook", f"no notebook {NBF} (the units kit ships the export)")


def main():
    for t in (t_plan, t_launch, t_switch_time, t_readback, t_misc, t_body_static, t_body_fakes, t_boot_fakes, t_main_fakes,
              t_window_fakes, t_facts):
        try:
            t()
        except Exception as e:
            check(t.__name__ + "_crashed", False, f"{type(e).__name__}: {e} | {traceback.format_exc()[-1500:]}")
    print(f"{RES['pass']}/{RES['total']} PASS", flush=True)
    sys.exit(0 if RES["pass"] == RES["total"] else 1)


if __name__ == "__main__":
    main()
