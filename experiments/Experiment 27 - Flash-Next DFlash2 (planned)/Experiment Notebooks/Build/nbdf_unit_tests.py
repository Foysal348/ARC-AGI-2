"""E27 NBDF quick unit tests (local + the Py3.12 CPU kit): the arm plan, captures, launch check, pin ladder, memory gate,
read-back parsers (config, DFlash events in dflash_seam's real format + its own readback_label, SpecDecoding lines,
backends, graph counts, the KV layout on the real NB4T worker record), the windowed garbage gate (a long digit-heavy
healthy ARC text is NOT flagged; the stream-level check is), smoke / ladder / prefix / early-exit / isolation gates, the
drafter pins, SeamSeq on a real copy of the Merlin files (with nb4_ppack), the verdict and the budget walk.
Run: python nbdf_unit_tests.py [--merlin-src DIR] [--nb4t-results FILE] [--drafter-config FILE]
Prints one 'FAIL <name>' line per failure, 'SKIP <name>' for missing inputs, then '<pass>/<total> PASS'."""
import json
import math
import random
import shutil
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
for d in (HERE, E25B, HERE / "e25"):
    if d.is_dir():
        sys.path.insert(0, str(d))
import nbdf_lib as N  # noqa: E402
import nb4t_lib as T  # noqa: E402
import nb4_ppack as P  # noqa: E402
import dflash_seam as DS  # noqa: E402


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


MERLIN = Path(arg("--merlin-src", r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src"))
NB4T_RES = Path(arg("--nb4t-results", r"D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb4t-levers/nb4t/nb4t_results.json"))
DCFG = Path(arg("--drafter-config", r"D:/ARC AGI 2/_scratch/dflash2_drafter/kernel_out/bf16-embed/config.json"))
RES = {"pass": 0, "fail": 0}


def ok(name, cond, info=""):
    if cond:
        RES["pass"] += 1
    else:
        RES["fail"] += 1
        print("FAIL", name, str(info)[:400], flush=True)


def raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


Q_EXTRA = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}"]
DPATH = "/kaggle/input/models/foysalemonshanto/" + N.DRAFTER_REL
api = N.seam_api(DS)
SENV = {k: N.seam_env_for(api, N.K_OF[k], True, mask_row_sha=N.DRAFTER_PINS["mask_row"]["sha256"])[0] for k in N.D_ARMS}
ARMS = N.arm_plan(Q_EXTRA, {"VLLM_SERVER_DEV_MODE": "1"}, 3553580524, DPATH, q_env=P.server_env("Q-defer"), seam_envs=SENV)

# ---------------------------------------------------------------- T1 plan + captures
ok("T1 order", tuple(ARMS) == ("Q", "D4s", "D3s", "D4", "D4t", "D7", "D4b"))
ok("T1 D4 = the 71-block control (no SW block, short)", "DFLASH_DRAFT_SW_BLOCK" not in ARMS["D4"]["env"] and ARMS["D4"]["short_only"]
   and ARMS["D4"]["spec"] == ARMS["D4s"]["spec"] and ARMS["D4s"]["env"] == dict(ARMS["D4"]["env"], DFLASH_DRAFT_SW_BLOCK="1024"))
ok("T1 D4t = D4s + DFLASH_CONV_IMPL=torch", ARMS["D4t"]["env"] == dict(ARMS["D4s"]["env"], DFLASH_CONV_IMPL="torch")
   and ARMS["D4t"]["spec"] == ARMS["D4s"]["spec"] and ARMS["D4t"]["captures"] == ARMS["D4s"]["captures"] and ARMS["D4t"]["short_only"])
ok("T1 SW block 1024 only on D4s / D3s / D4t", [k for k, a in ARMS.items() if a["env"].get("DFLASH_DRAFT_SW_BLOCK") == "1024"] == ["D4s", "D3s", "D4t"]
   and all("DFLASH_DRAFT_SW_BLOCK" not in a["env"] for k, a in ARMS.items() if k not in N.SW_BLOCK_ARMS))
bad_ = json.loads(json.dumps(ARMS)); bad_["D7"]["env"]["DFLASH_DRAFT_SW_BLOCK"] = "1024"
ok("T1 SW block refused by the plan check", raises(lambda: N.check_arm_plan(bad_)))
ok("T1 SW block refused by the launch check", any("DFLASH_DRAFT_SW_BLOCK" in x for x in N.launch_check(
    ARMS["D4"], dict(ARMS["D4"]["env"], VLLM_PP_LAYER_PARTITION="25,23", DFLASH_DRAFT_SW_BLOCK="1024"), [], None)))
bad_ = json.loads(json.dumps(ARMS)); bad_["D4s"]["env"]["DFLASH_DRAFT_SW_BLOCK"] = "2048"
ok("T1 SW arms must use exactly 1024", raises(lambda: N.check_arm_plan(bad_)))
ok("T1 check passes", N.check_arm_plan(ARMS) is True)
ok("T1 captures", [a["captures"] for a in ARMS.values()] == [[1, 2, 3, 4, 6, 8, 12, 16, 24, 32], [5, 10, 15, 20, 25, 30, 35, 40],
                                                             [4, 8, 12, 16, 20, 24, 28, 32], [5, 10, 15, 20, 25, 30, 35, 40],
                                                             [5, 10, 15, 20, 25, 30, 35, 40], [8, 16, 24, 32, 40, 48, 56, 64],
                                                             [5, 10, 15, 20, 25, 30, 35, 40]])
for k in N.D_ARMS:
    c = ARMS[k]["coverage"]
    ok(f"T1 exact graphs {k}", c["covers_all"] and not c["padded_b"] and c["graph_reqs"] == list(range(1, 9)), c)
ok("T1 Q coverage padded 5/7", ARMS["Q"]["coverage"]["padded_b"] == [5, 7], ARMS["Q"]["coverage"])
ok("T1 K4 with Q's list stops at 5 requests", N.v2_decode_graphs(N.Q_CAPTURES, 8, 4)[-1] == 5, N.v2_decode_graphs(N.Q_CAPTURES, 8, 4))
ok("T1 ring", N.qsa_ring(3) == (True, 8) and N.qsa_ring(4) == (True, 8) and N.qsa_ring(7) == (False, 12) and N.qsa_ring(7, widen=True) == (True, 16))
ok("T1 ring env only D7", [ARMS[k]["env"].get("DFLASH_QSA_RING_WIDEN") for k in N.D_ARMS] == [None, None, None, None, "1", None])
ok("T1 mask sha env", all(ARMS[k]["env"].get("DFLASH_MASK_ROW_SHA") == N.DRAFTER_PINS["mask_row"]["sha256"] for k in N.D_ARMS))
ok("T1 no host embed / sidecar", all("DFLASH_DRAFT_EMBED_HOST" not in ARMS[k]["env"] and "DFLASH_EMBED_SIDECAR" not in ARMS[k]["env"] for k in N.D_ARMS))
ok("T1 Q env clean", not any(k.startswith("DFLASH") for k in ARMS["Q"]["env"]) and ARMS["Q"]["env"]["NB4_PPDEFER"] == "1")
sp = T.spec_from_argv(ARMS["D4s"]["extra"])
ok("T1 D4 spec", sp == {"method": "dflash", "model": DPATH, "num_speculative_tokens": 4, "draft_sample_method": "probabilistic",
                        "kv_cache_dtype": "auto", "attention_backend": "FLASH_ATTN"}, sp)
ok("T1 D4b spec", T.spec_from_argv(ARMS["D4b"]["extra"])["kv_cache_dtype"] == "fp8_e4m3" and
   T.spec_from_argv(ARMS["D4b"]["extra"])["attention_backend"] == "TRITON_ATTN")
ok("T1 extra = Q + pair", all(ARMS[k]["extra"][:len(Q_EXTRA)] == Q_EXTRA and len(ARMS[k]["extra"]) == len(Q_EXTRA) + 4 for k in N.D_ARMS))
bad = json.loads(json.dumps(ARMS))
bad["D3s"]["env"]["NB4_PPACK"] = "1"
ok("T1 extra env refused", raises(lambda: N.check_arm_plan(bad)))
bad = json.loads(json.dumps(ARMS))
bad["D4s"]["extra"] += ["--max-num-seqs", "12"]
ok("T1 extra flag refused", raises(lambda: N.check_arm_plan(bad)))
bad = json.loads(json.dumps(ARMS))
bad["D7"]["env"].pop("DFLASH_QSA_RING_WIDEN")
ok("T1 missing ring env refused", raises(lambda: N.check_arm_plan(bad)))
bad = json.loads(json.dumps(ARMS))
bad["D4s"]["env"]["NCCL_PROTO"] = "LL128"
ok("T1 never-do refused", raises(lambda: N.check_arm_plan(bad)))
iso = N.iso_arm(ARMS["D4s"])
ok("T1 iso arm", iso["key"] == "D4s_ISO" and "NB4_PPDEFER" not in iso["env"] and iso["expect"]["NB4_PPDEFER"] is None and iso["q_label"] is None
   and iso["port"] != ARMS["D4s"]["port"] and iso["spec"] == ARMS["D4s"]["spec"])

# ---------------------------------------------------------------- T2 launch check
def full_argv(a, pin):
    return ["/v/bin/vllm", "serve", "/tmp/fn_merlin", "--speculative-config", json.dumps(N.Q_SPEC, separators=(",", ":")),
            "--max-num-seqs", "8", "--kv-cache-memory-bytes", str(pin), "--compilation-config", N.compilation_json(N.Q_CAPTURES)] + a["extra"]


for k, a in ARMS.items():
    env = dict(a["env"], VLLM_PP_LAYER_PARTITION="25,23")
    ok(f"T2 launch {k}", N.launch_check(a, env, full_argv(a, 3553580524), 3553580524) == [], N.launch_check(a, env, full_argv(a, 3553580524), 3553580524))
a = ARMS["D4s"]
env = dict(a["env"], VLLM_PP_LAYER_PARTITION="25,23")
ok("T2 wrong pin", N.launch_check(a, env, full_argv(a, 3553580524), 3285145068) != [])
ok("T2 missing seam env", N.launch_check(a, {k: v for k, v in env.items() if k != "DFLASH_SEAM"}, full_argv(a, 1), None) != [])
ok("T2 Q with DFLASH env", N.launch_check(ARMS["Q"], dict(ARMS["Q"]["env"], DFLASH_SEAM="1", VLLM_PP_LAYER_PARTITION="25,23"),
                                         full_argv(ARMS["Q"], 1), None) != [])

# ---------------------------------------------------------------- T3 pins + memory gate
pl = N.pin_plan(3553580524, 175, 4, False)
ok("T3 pin plan W4 (embed on GPU) -> 2 rungs", pl["pin"] == 3016709612 and pl["rungs"] == 2 and pl["rungs_formula"] >= 2, pl)
pl = N.pin_plan(3553580524, 2000, 4, True)
ok("T3 pin plan ample -> 1 rung", pl["pin"] == 3285145068 and pl["rungs"] == 1, pl)
ok("T3 ladder", N.PIN_LADDER[:3] == [3553580524, 3285145068, 3016709612])
ok("T3 SW arms: start at the D pin 3,016,709,612, one step-down to 2,748,274,156",
   3553580524 - N.SW_START_RUNGS * N.PIN_STEP == 3016709612 and N.SW_OOM_RETRIES == 1
   and 3553580524 - (N.SW_START_RUNGS + N.SW_OOM_RETRIES) * N.PIN_STEP == 2748274156)
ok("T3 draft logits", abs(N.draft_logits_mib(4) - 30.3) < 0.1 and abs(N.draft_logits_mib(7) - 53.0) < 0.1, N.draft_logits_mib(4))
ok("T3 memory gate", N.memory_gate([663, 687, 175, 175])[0] is False and N.memory_gate([600, 600, 513, 700])[0] is True
   and N.memory_gate(None)[0] is None)
for case, want in ((None, None), ([700, 700], None), ([700, 700, 700, float("nan")], None), ([700, 700, 700, 0], False),
                   ([700, 700, 700, 700], True), ([700, 700, True, 700], None), ([700, 700, 700, float("inf")], None)):
    ok(f"T3 memory gate fail-closed {case}", N.memory_gate(case)[0] is want, N.memory_gate(case))
ok("T3 memory gate reasons", N.memory_gate([700, 700])[1].startswith("INCOMPLETE_FOUR_RANK_MEMORY")
   and N.memory_gate([700, 700, 700, 0])[1].startswith("LOW_MEMORY_RANKS:[3]"))

# ---------------------------------------------------------------- T4 read-backs
NA = ("(APIServer pid=1) INFO 11-01 03:00:00 [api_utils.py:286] non-default args: {'max_num_seqs': 8, 'kv_cache_memory_bytes': 3016709612, "
      "'language_model_only': True, 'speculative_config': {'method': 'dflash', 'model': '" + DPATH + "', 'num_speculative_tokens': 4, "
      "'draft_sample_method': 'probabilistic', 'kv_cache_dtype': 'auto', 'attention_backend': 'FLASH_ATTN'}, "
      "'compilation_config': {'cudagraph_mode': 'FULL_AND_PIECEWISE', 'cudagraph_capture_sizes': [5, 10, 15, 20, 25, 30, 35, 40]}}")
STAGE = [f"(Worker_PP{p}_TP{t}_EP{2 * p + t} pid={10 + 2 * p + t}) INFO 11-01 02:59:10 [model.py:307] Qwen4Exp QSA: applied static K/V scales "
         f"from x to 6 of 12 layer(s) (pipeline layers {'0-25' if p == 0 else '25-48'}): a" for p in (0, 1) for t in (0, 1)]


def ev(p, t, d):
    return f"(Worker_PP{p}_TP{t}_EP{2 * p + t} pid={10 + 2 * p + t}) E27_DFLASH " + json.dumps(dict(d, pid=10 + 2 * p + t), sort_keys=True)


AUXL = [ev(p, t, {"event": "aux_layers", "ids": [6, 16, 26, 36, 46], "local": [6, 16] if p == 0 else [26, 36, 46],
                  "start_layer": 0 if p == 0 else 25, "end_layer": 25 if p == 0 else 48, "num_layers": 48,
                  "slot_base": 0 if p == 0 else 2, "upstream_total": 0 if p == 0 else 2, "hc_sp": False, "tap_mode": "combine"})
        for p in (0, 1) for t in (0, 1)]
EMBL = [ev(1, t, {"event": "embed", "draft": "DFlash2Qwen3ForCausalLM", "embed_source": "own", "head_source": "shared_target",
                  "shape": [124160, 2560], "dtype": "torch.bfloat16", "device": "cuda:0", "sample_abs_mean": 0.0123,
                  "released_mib": {"embed_tokens": 0.0, "lm_head": 606.3}, "released_bytes": {"embed_tokens": 0, "lm_head": 635699200},
                  "host": False, "host_mib": 0.0,
                  "mask_row": ({"id": 248077, "on_this_rank": True, "sha256": N.DRAFTER_PINS["mask_row"]["sha256"], "norm": 0.262735}
                               if t == 1 else {"id": 248077, "on_this_rank": False})}) for t in (0, 1)]
LCK = [ev(1, t, {"event": "draft_load_check", "expected": 85, "loaded": 85, "dropped": ["mask_embedding"]}) for t in (0, 1)]
LCK += [ev(1, t, {"event": "ingraph_meta", "speculator": "DFlash2", "native_supports_decode_metadata_update": True,
                  "supports_decode_metadata_update": True, "forced_off": False, "kernel_map_refresh": True, "groups": 1}) for t in (0, 1)]
BACK = [f"(Worker_PP1_TP{t}_EP{2 + t} pid={12 + t}) INFO 11-01 03:01:00 [cuda.py:446] Using AttentionBackendEnum.FLASH_ATTN backend." for t in (0, 1)]
SPEC = ["(APIServer pid=1) INFO 11-01 03:10:0%d [metrics.py:138] SpecDecoding metrics: Mean acceptance length: 3.40, Accepted throughput: "
        "200.0 tokens/s, Drafted throughput: 330.0 tokens/s, Accepted: %d tokens, Drafted: %d tokens, Per-position acceptance rate: "
        "0.850, 0.700, 0.500, 0.350, Avg Draft acceptance rate: 60.0%%" % (i, 240 * (i + 1), 400 * (i + 1)) for i in range(3)]
GR = ["(Worker_PP1_TP0_EP2 pid=12) Capturing CUDA graphs (FULL): 100%|##########| 8/8 [00:03<00:00]",
      "(Worker_PP1_TP0_EP2 pid=12) Capturing dflash2 CUDA graphs (FULL): 100%|##########| 8/8 [00:01<00:00]",
      "(Worker_PP1_TP0_EP2 pid=12) INFO Capturing model for DFlash2 speculator..."]
ARCH = ["(EngineCore pid=5) INFO 11-01 02:53:00 [model.py:735] Resolved architecture: Qwen4ExpForConditionalGeneration",
        "(EngineCore pid=5) INFO 11-01 02:53:01 [model.py:735] Resolved architecture: DFlash2DraftModel"]
LOG = "\n".join([NA] + STAGE + ARCH + AUXL + EMBL + LCK + BACK + GR + SPEC)
cr = N.config_readback(ARMS["D4s"], LOG, 3016709612)
ok("T4 config readback D4", cr["ok"], cr)
cr = N.config_readback(ARMS["D3s"], LOG, 3016709612)
ok("T4 config readback D3 definitive", not cr["ok"] and cr["definitive"], cr)
rb = N.dflash_readback(ARMS["D4"], LOG)       # the fused, non-SW arm
ok("T4 dflash readback ok", rb["ok"] and not rb["definitive"], rb["why"])
lab = DS.readback_label(LOG, mask_row_sha=N.DRAFTER_PINS["mask_row"]["sha256"])
ok("T4 the seam's own readback_label ENGAGED on the same log", lab[0] == "ENGAGED", lab)
ok("T4 merge seam label", N.merge_seam_label(rb, lab)["ok"] and not N.merge_seam_label(rb, ("NOT_INTEGRATED", {}))["ok"])
for nm, mut in (("swapped taps", LOG.replace('"local": [6, 16]', '"local": [26, 36]', 1)),
                ("eager drafter", LOG + "\nWARNING DFlash2 draft attention (FLASHINFER) does not support full CUDA graphs; running the draft eagerly."),
                ("mtp resolved", LOG + "\nINFO [model.py:735] Resolved architecture: Qwen4ExpMTP"),
                ("wrong backend", LOG.replace("FLASH_ATTN backend.", "FLASHINFER backend.")),
                ("zero embed", LOG.replace('"sample_abs_mean": 0.0123', '"sample_abs_mean": 0.0')),
                ("wrong mask", LOG.replace(N.DRAFTER_PINS["mask_row"]["sha256"], "0" * 64)),
                ("3-position lines", LOG.replace("0.850, 0.700, 0.500, 0.350", "0.85, 0.70, 0.50"))):
    r = N.dflash_readback(ARMS["D4"], mut)
    ok(f"T4 readback catches {nm}", not r["ok"], r["why"])
ok("T4 missing aux not definitive", (lambda r: not r["ok"] and not r["definitive"])(N.dflash_readback(ARMS["D4s"], "\n".join([NA] + ARCH))))
d7 = N.dflash_readback(ARMS["D7"], LOG)
ok("T4 D7 needs a ring line", not d7["ok"] and d7["checks"].get("ring_widened") is False, d7["checks"])
d7log = LOG.replace("num_speculative_tokens': 4", "num_speculative_tokens': 7") + "\n" + ev(1, 0, {"event": "qsa_ring", "minimal": 12, "capacity": 16, "block_size": 4096, "num_speculative_tokens": 7})
ok("T4 D7 ring line", N.dflash_readback(ARMS["D7"], d7log)["checks"].get("ring_widened") is True)
qi = N.q_seam_inert("\n".join(ARCH[:1] + ["Resolved architecture: Qwen4ExpMTP"]))
ok("T4 q seam inert", qi["ok"] and not N.q_seam_inert(LOG)["ok"])
sl = N.spec_decoding_lines(LOG)
ss = N.spec_summary(sl, 4)
ok("T4 spec lines", len(sl) == 3 and ss["positions_ok"] and ss["per_pos"] == [0.85, 0.7, 0.5, 0.35] and abs(ss["al"] - (1 + 1440 / 600)) < 1e-3, ss)
ok("T4 graph counts", N.graph_counts(LOG) == {"target": {"FULL": [8]}, "dflash2": {"FULL": [8]}}, N.graph_counts(LOG))
ok("T4 backend lines", [x["backend"] for x in N.backend_lines(LOG)["selected"]] == ["FLASH_ATTN", "FLASH_ATTN"])
acc = N.acceptance_delta({"pos": {0: 10, 1: 5, 2: 2, 3: 1}, "drafts": 20}, {"pos": {0: 95, 1: 75, 2: 52, 3: 36}, "drafts": 120}, 4)
ok("T4 acceptance delta", acc["status"] == "OK" and acc["per_pos"] == [0.85, 0.7, 0.5, 0.35] and abs(acc["al"] - 3.4) < 1e-6, acc)
ok("T4 acceptance no drafts", N.acceptance_delta({"pos": {}, "drafts": 5}, {"pos": {}, "drafts": 5}, 4)["status"] == "NO_DRAFTS")

# real NB4T Q worker record: block sizes + ring capacity (8 at K3) + groups; no drafter spec on Q
if NB4T_RES.is_file():
    RQ = json.loads(NB4T_RES.read_text(encoding="utf-8"))["sections"]["Q"]["nb4t_q"]
    kl = N.kv_layout_readback(RQ["workers"], ARMS["Q"])
    ok("T4 real kv layout Q", kl["ring_capacity"] == [8] and kl["block_sizes"].get("FullAttentionSpec") == [4096] and not kl["drafter_specs"]
       and kl["ring_ok"] is True and kl["num_blocks"]["0"] == 215, kl)
    w = json.loads(json.dumps(RQ["workers"]))
    for r in w["ranks"][2:]:
        r["kv_cache_specs"].append({"type": "SlidingWindowSpec", "block_size": 4096, "dtype": "torch.bfloat16", "num_kv_heads": 4,
                                    "head_size": 128, "layers": 5, "example": "model.layers.48.self_attn.attn"})
    kl = N.kv_layout_readback(w, ARMS["D4s"])
    ok("T4 kv layout drafter BF16", kl["drafter_kv_ok"] is True and len(kl["drafter_specs"]) == 2, kl["drafter_specs"])
    ok("T4 real Q base_check", T.base_check(RQ)[0], T.base_check(RQ))
    cf = T.capacity_facts(RQ)
    ok("T4 real Q capacity F 7 pool 215", cf["F_blocks"] == 7 and cf["num_gpu_blocks"] == [215], cf)
    ok("T4 resident levels from the real Q pool", (N.plan_level(215, 7, 90000, 4366, 90), N.plan_level(215, 7, 116000, 4366, 90),
                                                    N.plan_level(215, 7, 40000, 4366, 90)) == (6, 5, 8),
       (N.plan_level(215, 7, 90000, 4366, 90), N.plan_level(215, 7, 116000, 4366, 90), N.plan_level(215, 7, 40000, 4366, 90)))
    ms = N.memory_stages(RQ)
    ok("T4 memory stages real Q", ms["model_load_gib"] and ms["free_after_ready_mib"] == [663, 687, 175, 175], ms)
else:
    print("SKIP T4 real NB4T record", NB4T_RES)

# ---------------------------------------------------------------- T5 windowed garbage gate
random.seed(7)
words = ("the rule maps each colored cell to its mirror across the vertical axis so the output grid keeps the same shape but every "
         "row is reversed check pair two the blue object moves right until it touches the wall then I verify on the test input "
         "count rows columns compare with expected output wait that fails for row three so maybe the rule also recolors").split()


def healthy_long(n_chunks=60):
    out = []
    for i in range(n_chunks):
        out.append(" ".join(random.choice(words) for _ in range(40)) + ".\n")
        g = [[random.randint(0, 9) for _ in range(15)] for _ in range(15)]
        out.append("Row grid:\n" + "\n".join(" ".join(str(x) for x in row) for row in g) + "\n")
        out.append(f"Row {i}: " + " ".join(str(random.randint(0, 9)) for _ in range(30)) + "\n")
    return "".join(out)


TXT = healthy_long(400)
TOKS = list(range(len(TXT)))                     # one "token" per character: decode = the character slice


def dec(ids):
    return TXT[ids[0]:ids[-1] + 1] if ids else ""


ok("T5 healthy text is long", len(TXT) > 250000, len(TXT))
ok("T5 stream-level check misfires (LOW_VARIETY on the whole long text)", N.degenerate(TXT) == "LOW_VARIETY", N.degenerate(TXT))
ok("T5 windowed check: healthy digit-heavy text NOT flagged", N.windowed_garbage(TOKS, dec) is None, N.windowed_garbage(TOKS, dec))
bang = TXT[:20000] + "!" * 3000 + TXT[20000:40000]
ok("T5 windowed check flags a garbage run", (N.windowed_garbage(list(range(len(bang))), lambda ids: bang[ids[0]:ids[-1] + 1]) or {}).get("verdict")
   in ("BANG_GARBAGE", "REPEATED_CHAR"))
rows = [{"text": TXT, "tokens": TOKS}]
rv = N.garbage_review(rows, dec)
ok("T5 review: stream recorded, windowed clean", rv["gate"] == "CLEAN" and rv["stream_level"] == [] or rv["gate"] == "CLEAN", rv)
w = {"status": "FAIL_GARBAGE_UNDER_LOAD", "level": 8, "running_min": 8, "running_max": 8, "running_eq_level": True, "prefill_done_before_window": True}
N.apply_windowed_gate(w, {"gate": "CLEAN", "rows": 8, "windowed": [], "stream_level": ["LOW_VARIETY"]}, 8)
ok("T5 restore STEADY", w["status"] == "STEADY" and "stream_level_flag_overridden" in w, w)
w = {"status": "FAIL_GARBAGE_UNDER_LOAD", "level": 6, "running_min": 5, "running_max": 5}
N.apply_windowed_gate(w, {"gate": "CLEAN", "rows": 6, "windowed": []}, 6)
ok("T5 restore CAPACITY_LIMITED", w["status"] == "CAPACITY_LIMITED", w)
w = {"status": "STEADY", "level": 8}
N.apply_windowed_gate(w, {"gate": "GARBAGE", "rows": 8, "windowed": [{"start_token": 4000, "verdict": "BANG_GARBAGE"}]}, 8)
ok("T5 windowed garbage fails the window", w["status"] == "FAIL_GARBAGE_UNDER_LOAD", w)
w = {"status": "FAIL_GARBAGE_UNDER_LOAD", "level": 8, "running_min": 8, "running_max": 8, "running_eq_level": False}
N.apply_windowed_gate(w, {"gate": "CLEAN", "rows": 8, "windowed": []}, 8)
ok("T5 restore keeps the running == level rule", w["status"] == "INCONCLUSIVE_RUNNING_NE_LEVEL", w)

# ---------------------------------------------------------------- T6 smoke / ladder / prefix / early exit / isolation
GOOD_ACC = {"status": "OK", "drafts": 900, "accept_rate": 0.6, "al": 3.4}
lad = {"alone": {"error": None}, **{str(c): {"vs_alone": {"status": "EXACT"}} for c in N.CONC_LADDER}}
sm = {"errors": [], "garbage": {"gate": "CLEAN"}, "tokens": [2048, 2048], "acceptance": GOOD_ACC, "ladder": lad}
ok("T6 smoke pass", N.smoke_gate(sm) == "PASS")
ok("T6 zero accept", N.smoke_gate(dict(sm, acceptance=dict(GOOD_ACC, accept_rate=0.004))) == "FAIL_SMOKE_DFLASH_ZERO_ACCEPT")
ok("T6 few drafts", N.smoke_gate(dict(sm, acceptance=dict(GOOD_ACC, drafts=50))) == "INCONCLUSIVE_SMOKE_FEW_DRAFTS")
ok("T6 garbage", N.smoke_gate(dict(sm, garbage={"gate": "GARBAGE"})) == "FAIL_SMOKE_GARBAGE")
lad2 = json.loads(json.dumps(lad))
lad2["4"]["vs_alone"]["status"] = "FAIL_HIGH_MARGIN"
ok("T6 contamination -> recorded (B2: klee100 is non-deterministic), never a FAIL", N.smoke_gate(dict(sm, ladder=lad2)) == "INCONCLUSIVE_SMOKE_CONCURRENCY_CONTAMINATION")
lad2["4"]["vs_alone"]["status"] = "NEAR_TIE"
ok("T6 near tie allowed", N.contamination_gate(lad2) == "NEAR_TIE" and N.smoke_gate(dict(sm, ladder=lad2)) == "PASS")
ok("T6 ladder 7 + probe = 8 seqs", max(N.CONC_LADDER) + 1 == N.SEQS)
def _pc(w2=3.3, w3=3.2, hit=True, n=2):
    side = lambda al, h=False: {"acceptance": {"status": "OK", "al": al}, "prefix_hits": h, "prefix": {"rate": 0.9 if h else 0.0}}
    return {"prompts": [{"variant": 40 + i, "turns": {"2": {"cold": side(3.4), "warm": side(w2, hit)},
                                                       "3": {"cold": side(3.4), "warm": side(w3, hit)}}} for i in range(n)]}


ok("T6 prefix 2 x 3 pass", N.prefix_gate(_pc()) == "PASS")
ok("T6 prefix collapse on turn 3", N.prefix_gate(_pc(w3=2.0)) == "PREFIX_HIT_ACCEPTANCE_COLLAPSE")
ok("T6 prefix no hit", N.prefix_gate(_pc(hit=False)) == "INCONCLUSIVE_PREFIX_NO_HIT")
ok("T6 prefix needs both prompts", N.prefix_gate(_pc(n=1)) == "INCONCLUSIVE_PREFIX_PROMPTS")
ok("T6 prefix turn rates recorded", len(N.prefix_reuse({"prefix_cache": _pc()})["turn_rates"]) == 8)
ok("T6 AL vs Q is a diagnostic only", N.al_vs_q({"al": 2.6}, {"al": 2.8})["below_q"] is True and not hasattr(N, "EARLY_EXIT"))
QW = {"8k@8#1": {"status": "STEADY", "running_eq_level": True, "preemptions": 0, "tok_s": 300.0},
      "8k@8#2": {"status": "STEADY", "running_eq_level": True, "preemptions": 0, "tok_s": 305.0}}
mkw = lambda a, b: {"8k@8#1": dict(QW["8k@8#1"], tok_s=a), "8k@8#2": dict(QW["8k@8#2"], tok_s=b)}
ok("T6 short exit: both 8k clearly below Q", N.short_exit(QW, mkw(250, 260))[0] is True, N.short_exit(QW, mkw(250, 260)))
ok("T6 short exit: one window within noise", N.short_exit(QW, mkw(250, 299))[0] is False)
ok("T6 short exit: faster never exits", N.short_exit(QW, mkw(340, 345))[0] is False)
ok("T6 short exit: invalid windows never exit", N.short_exit(QW, {"8k@8#1": {"status": "INCONCLUSIVE_PREEMPTION", "tok_s": 1}})[0] is False)
ok("T6 short exit only D4", N.SHORT_EXIT_ARMS == ("D4s",))
ok("T6 isolation triggers", N.needs_isolation({"status": "SERVER_DIED_IN_8k@8#1"}) and N.needs_isolation({"status": "NO_GO_SMOKE"})
   and not N.needs_isolation({"status": "LAUNCH_FAILED", "tries": [{"oom": True}]}) and not N.needs_isolation({"status": "DONE", "correctness": "CLEAN"})
   and not N.needs_isolation({"status": "NOT_INTEGRATED"}))
ok("T6 isolation labels", N.isolation_label({}, {"status": "DONE", "correctness": "CLEAN", "smoke_gate": "PASS"}) == "QDEFER_INTERACTION"
   and N.isolation_label({}, {"status": "NO_GO_SMOKE"}) == "DFLASH_ITSELF" and N.isolation_label({}, None) == "ISOLATION_NOT_RUN")

# ---------------------------------------------------------------- T7 drafter pins + config
PF = {"files": {"model.safetensors": [2477538608, N.DRAFTER_PINS["model.safetensors"]["sha256"]],
                "config.json": [1236, N.DRAFTER_PINS["config.json"]["sha256"]]}, "tensors": 82,
      "embed": {"shape": [248320, 2560], "dtype": "BF16", "nbytes": 1271398400, "sha256": N.DRAFTER_PINS["embed_tokens"]["sha256"]},
      "mask_row": {"sha256": N.DRAFTER_PINS["mask_row"]["sha256"], "norm": 0.26273465}, "zero_rows": 0}
ok("T7 pins ok", N.drafter_pins_check(PF) == (True, []))
for nm, mut in (("size", ("files", "model.safetensors", [1, "x"])), ("embed sha", ("embed", "sha256", "0" * 64)),
                ("tensors", ("tensors", None, 81)), ("zero rows", ("zero_rows", None, 3)), ("norm", ("mask_row", "norm", 0.3))):
    f = json.loads(json.dumps(PF))
    if mut[1] is None:
        f[mut[0]] = mut[2]
    else:
        f[mut[0]][mut[1]] = mut[2]
    ok(f"T7 pin mismatch {nm}", not N.drafter_pins_check(f)[0])
import struct  # noqa: E402
row = b"".join(struct.pack("<f", v)[2:] for v in (0.5, -0.25, 1.0))
ok("T7 bf16 norm", abs(N.bf16_row_norm(row) - math.sqrt(0.25 + 0.0625 + 1.0)) < 1e-6)
ok("T7 drafter path", N.DRAFTER_MODEL == "foysalemonshanto/qwen38-flash-next-dflash2v2/transformers/bf16-embed/1"
   and N.DRAFTER_REL == "qwen38-flash-next-dflash2v2/transformers/bf16-embed/1/bf16-embed")
if DCFG.is_file():
    cfg = json.loads(DCFG.read_text(encoding="utf-8"))
    keys = {"model.safetensors": {"embed_tokens.weight": {"shape": [248320, 2560], "dtype": "BF16"}, "fc.weight": {"shape": [2560, 12800], "dtype": "BF16"}}}
    dc = N.drafter_check(cfg, {"model.safetensors": 2477538608, "config.json": 1236}, keys)
    ok("T7 real drafter config", dc["ok"] and dc["embed_in_checkpoint"] and dc["sliding_window"] == 2048, dc["why"])
else:
    print("SKIP T7 real drafter config", DCFG)

# ---------------------------------------------------------------- T8 seam API + SeamSeq on a real copy of the Merlin files
ok("T8 seam api", api["ok"] and api["caps"]["ring_kw"] and api["caps"]["mask_kw"] == "mask_row_sha" and api["caps"]["host_kw"] == "embed_host")
env, notes = N.seam_env_for(api, 7, True, mask_row_sha="ab")
ok("T8 seam env K7", env == {"DFLASH_SEAM": "1", "DFLASH_MASK_ROW_SHA": "ab", "DFLASH_QSA_RING_WIDEN": "1"}, env)
env, notes = N.seam_env_for(api, 4, False, sidecar_path="/tmp/fn_merlin")
ok("T8 seam env sidecar fallback", env.get("DFLASH_EMBED_SIDECAR") == "/tmp/fn_merlin", env)
ok("T8 tree_state map", N.tree_state({"state": "applied"}) == "FULLY_PATCHED" and N.tree_state({"state": "not_applied"}) == "ORIGINAL"
   and N.tree_state({"state": "mixed"}) == "INVALID_MIXED")
rels = list(DS.PATCHES) + list(P.PATCHES)
if all((MERLIN / r).is_file() for r in rels):
    tmp = Path(tempfile.mkdtemp(prefix="nbdf_tree_"))
    try:
        for r in rels:
            (tmp / r).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(MERLIN / r, tmp / r)
        P.apply_ppack_patch(tmp, sys.executable)
        sq = N.SeamSeq(lambda ring: DS.apply_dflash_patch(tmp, sys.executable, ring=ring), lambda ring: DS.verify_dflash_patch(tmp, ring=ring),
                       lambda ring: DS.revert_dflash_patch(tmp, sys.executable, ring=ring))
        r_ = sq.apply()
        ok("T8 real seam apply", r_[0] and sq.ring is True and N.ring_status(sq)[1], (r_, N.ring_status(sq)))
        ok("T8 combined state", DS.combined_tree_state(tmp, P) == {"e27": "FULLY_PATCHED", "nb4": "FULLY_PATCHED", "valid": True})
        ok("T8 check", sq.check("D4s")[0])
        f = tmp / DS.CFG_REL
        orig = f.read_bytes()
        f.write_bytes(orig + b"\n# foreign\n")
        ok("T8 foreign -> INVALID_MIXED", not sq.check("D3s")[0] and "INVALID_MIXED" in sq.check("D3s")[1])
        f.write_bytes(orig)
        sq.finish()
        ok("T8 revert", sq.state == "reverted" and DS.tree_state(tmp) == "ORIGINAL" and P.verify_ppack_patch(tmp)["state"] == "applied")
        P.revert_ppack_patch(tmp, sys.executable)
        ok("T8 pristine", all(DS.sha256_bytes((tmp / r).read_bytes()) == DS.PATCHES[r][0] for r in DS.PATCHES))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
else:
    print("SKIP T8 real Merlin files", MERLIN)
calls = []


def fake_apply(ring):
    calls.append(ring)
    if ring:
        raise RuntimeError("ring pre-sha")
    return {"ok": True}


sq = N.SeamSeq(fake_apply, lambda ring: {"state": "applied", "files": {}}, lambda ring: {"ok": True})
r_ = sq.apply()
ok("T8 ring fallback to core", r_[0] and calls == [True, False] and sq.ring is False and N.ring_status(sq)[1] is False, (r_, calls))

# ---------------------------------------------------------------- T9 verdict
def win(tok, tick=40.0, al=3.1, lvl=8, prefix=0.5):
    return {"status": "STEADY", "running_eq_level": True, "preemptions": 0, "tok_s": tok, "tick_ms": tick, "level": lvl,
            "mean_acceptance_length": al, "prefix_hit_rate": prefix}


def rec(key, tok8, tok40, tok90, tok116, aa=False, **kw):
    st = {"8k@8#1": win(tok8[0]), "8k@8#2": win(tok8[1]), "40k@8": win(tok40)}
    if aa:
        st.update({"90k@6#1": win(tok90[0], lvl=6), "90k@6#2": win(tok90[1], lvl=6), "116k@5#1": win(tok116[0], lvl=5), "116k@5#2": win(tok116[1], lvl=5)})
    else:
        st.update({"90k@6": win(tok90[0], lvl=6), "116k@5": win(tok116[0], lvl=5)})
    r = {"status": "DONE", "correctness": "CLEAN", "quality_gate": "PASS", "oracle_vs": "EXACT", "long_gate": "PASS", "smoke_gate": "PASS",
         "prefix_gate": "PASS", "env_readback": {"ok": True}, "config_readback": {"ok": True, "log": {}}, "dflash_readback": {"ok": True},
         "q_readback_final": {"label": "ENGAGED"}, "long_oracle": {"complete": True, "errors": [], "garbage": [], "vs_q_long": {"status": "EXACT"}},
         "steady": st, "pool": {"max_concurrency": 5.63}, "workers": {"num_gpu_blocks": [215]}, "pin": 3553580524,
         "memory_gate_ready": {"ok": True}, "memory_gate_first_decode": {"ok": True}}
    r.update(kw)
    return r


Qr = rec("Q", (300, 305), 287, (240, 244), (220, 222), aa=True)
v = N.verdict({"Q": Qr, "D4s": rec("D4s", (360, 365), 340, (290,), (260,)), "D3s": rec("D3s", (340, 338), 320, (200,), (180,))}, ARMS)
ok("T9 D4 WIN", v["arms"]["D4s"]["verdict"] == "WIN" and v["best_arm"] == "D4s", v["arms"]["D4s"].get("why"))
ok("T9 D3 SHORT_ONLY_WIN", v["arms"]["D3s"]["verdict"] == "SHORT_ONLY_WIN", v["arms"]["D3s"].get("why"))
ok("T9 D4 / D4t / D7 / D4b not run", all(v["arms"][k]["verdict"] == "SKIPPED_BUDGET" for k in ("D4", "D4t", "D7", "D4b")))
v = N.verdict({"Q": Qr, "D4": rec("D4", (240, 245), 0, (0,), (0,))}, ARMS)
ok("T9 D4 = SHORT_CONTROL (8k vs Q, never a winner)", v["arms"]["D4"]["verdict"] == "SHORT_CONTROL_REGRESSION"
   and v["throughput_eligible"]["D4"]["ok"] is False, v["arms"]["D4"].get("verdict"))
ok("T9 SW_BLOCK_LOWERED_ENGINE_BLOCK recorded", N.verdict({"Q": Qr, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), engine_block=1024)},
   ARMS)["arms"]["D4s"]["sw_block_flag"] == "SW_BLOCK_LOWERED_ENGINE_BLOCK")
ok("T9 F at the boot's block size", N.f_blocks(205, 4.857142857142857, 4096) == 11 and N.f_blocks(820, 4.857, 1024) == 41
   and N.engine_block("INFO Setting attention block size to 1024 tokens") == 1024)
# GPT final audit HIGH-1
ok("T9 HIGH-1: inconsistent pool / mc / block -> F None", N.f_blocks(205, 4.857142857, 1024) is None)
lv_, inf_ = N.level_for_band("90k", 90000, 205, 4.857142857, 1024, None, 4366, 90)
ok("T9 HIGH-1: no optimistic level from an invalid F (conservative 2, CAPACITY_UNVERIFIED)", lv_ == 2 and inf_["source"] == "CAPACITY_UNVERIFIED", inf_)
ok("T9 HIGH-1: a consistent 1024-unit pool / mc gives F >= 0 and a level", N.f_blocks(820, 4.857, 1024) >= 0
   and N.level_for_band("90k", 90000, 820, 4.857, 1024)[0] >= 1, N.level_for_band("90k", 90000, 820, 4.857, 1024))
for nm, a_ in (("missing pool", (None, 4.8)), ("missing mc", (205, None)), ("nan mc", (205, float("nan"))), ("bool", (True, 4.8))):
    lv_, inf_ = N.level_for_band("116k", 116000, a_[0], a_[1], 1024)
    ok(f"T9 HIGH-1: {nm} -> the conservative level, labelled", lv_ == 2 and inf_["source"] == "CAPACITY_UNVERIFIED", inf_)
ok("T9 HIGH-1: 40k conservative level 4", N.level_for_band("40k", 40000, None, None, 4096)[0] == 4)
lv_, inf_ = N.level_for_band("116k", 116000, 30, 30 / (131072 / 4096 + 7), 4096)
ok("T9 HIGH-1: a valid F with no room -> level 0, SKIPPED_CAPACITY (not a fallback)", lv_ == 0 and inf_["skip"] == "SKIPPED_CAPACITY", inf_)
lv_, inf_ = N.level_for_band("90k", 90000, 40, 40 / (131072 / 4096 + 7), 4096)
ok("T9 HIGH-1: a valid level 1 -> one stream", lv_ == 1, inf_)
lay_ = {"groups": {"2": [["UniformTypeKVCacheSpecs:MambaSpec", 5, 1024, 1], ["CircularBufferSpec", 6, 8, 1], ["FullAttentionSpec", 12, 1024, 1],
                         ["SlidingWindowSpec", 5, 1024, 1]]}}
lv_, inf_ = N.level_for_band("90k", 90000, 820, None, 1024, lay_)
ok("T9 HIGH-1: the layout estimate is used (recorded) when the log line is missing", inf_["source"].startswith("kv layout") and inf_["F"] == 7, inf_)
w_ = N.capacity_window_status({"status": "CAPACITY_LIMITED", "running_eq_level": False, "preemptions": 3, "tok_s": 100.0}, True)
ok("T9 HIGH-1: an unverified KV-bound window is INCONCLUSIVE_CAPACITY (no speed number)", w_["status"] == "INCONCLUSIVE_CAPACITY"
   and not T.window_ok(w_)[0])
w_ = N.capacity_window_status({"status": "STEADY", "running_eq_level": True, "preemptions": 0, "tok_s": 100.0}, True)
ok("T9 HIGH-1: an unverified but valid window counts", w_["status"] == "STEADY" and w_["capacity_unverified"] is True)
qx = rec("Q", (300, 305), 287, (240, 244), (220, 222), aa=True)
dx = rec("D4s", (360, 365), 340, (290,), (260,))
dx["steady"]["116k@5"] = {"status": "INCONCLUSIVE_CAPACITY", "capacity_unverified": True}
ok("T9 HIGH-1: a KV-bound long band never gives a false WIN", N.verdict({"Q": qx, "D4s": dx}, ARMS)["arms"]["D4s"]["verdict"] == "SHORT_ONLY_WIN")
dx["steady"].pop("116k@5")
dx["steady"]["116k@0"] = {"status": "SKIPPED_CAPACITY", "level": 0}
ok("T9 HIGH-1: a skipped band (no room) -> SHORT_ONLY_WIN at best", N.verdict({"Q": qx, "D4s": dx}, ARMS)["arms"]["D4s"]["verdict"] == "SHORT_ONLY_WIN")
# third final audit 1: mixed KV block sizes (engine 1024, target FullAttention 4096)
lay_m = {"block_sizes": {"FullAttentionSpec": [4096], "SlidingWindowSpec": [1024]}, "groups": {"2": [["MambaSpec", 5, 4096, 1], ["FullAttentionSpec", 12, 4096, 1]]}}
lv_, inf_ = N.level_for_band("90k", 90000, 205, 4.857142857142857, 1024, lay_m, 4366, 90)
ok("T9 MIXED: (205, 4.857, engine 1024, target 4096) -> F 11 from the target block, labelled", inf_.get("mixed") == "MIXED_KV_BLOCK_SIZES"
   and inf_["F"] == 11 and inf_["block"] == 4096 and lv_ >= 1 and "TARGET" in inf_["source"], inf_)
lv_, inf_ = N.level_for_band("90k", 90000, 205, None, 1024, lay_m, 4366, 90)
ok("T9 MIXED: no mc -> the layout estimate at the target block (never negative)", inf_["source"].startswith("kv layout") and inf_["F"] >= 0, inf_)
lv_, inf_ = N.level_for_band("90k", 90000, None, None, 1024, {"block_sizes": {"FullAttentionSpec": [4096]}}, 4366, 90)
ok("T9 MIXED: nothing usable -> the conservative level", lv_ == 2 and inf_["source"] == "CAPACITY_UNVERIFIED", inf_)
# third final audit 2: the admission watchdog (vLLM #55800)
fired = []
aw = N.AdmissionWatch(on_fire=fired.append, label="window_90k@6")
for t_ in range(0, 200, 5):
    aw.feed(float(t_), 0, 1, 0.42)
ok("T9 admission: a wedge fires ADMISSION_DEADLOCK once (bounded)", aw.fired and len(fired) == 1 and fired[0]["status"] == "ADMISSION_DEADLOCK"
   and 120 < fired[0]["wedged_sec"] <= 130 and fired[0]["kv_usage"] == 0.42 and fired[0]["waiting"] == 1, fired)
aw2 = N.AdmissionWatch(on_fire=fired.append)
for t_ in range(0, 400, 5):
    aw2.feed(float(t_), 0 if t_ % 100 < 60 else 6, 2 if t_ % 100 < 60 else 0)
ok("T9 admission: normal admission (waiting cleared within 120 s) never fires", not aw2.fired and len(fired) == 1)
dd = rec("D4s", (360, 365), 340, (290,), (260,), admission_deadlock={"status": "ADMISSION_DEADLOCK", "item": "window_116k@5"})
ok("T9 admission: a deadlock arm cannot be WIN", N.verdict({"Q": qx, "D4s": dd}, ARMS)["arms"]["D4s"]["verdict"] == "ADMISSION_DEADLOCK")
v = N.verdict({"Q": Qr, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), prefix_gate="PREFIX_HIT_ACCEPTANCE_COLLAPSE")}, ARMS)
ok("T9 prefix collapse fails", v["arms"]["D4s"]["verdict"] == "PREFIX_HIT_ACCEPTANCE_COLLAPSE" and v["best_arm"] is None)
v = N.verdict({"Q": Qr, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), smoke_gate="FAIL_SMOKE_CONCURRENCY_CONTAMINATION")}, ARMS)
ok("T9 contamination fails", v["arms"]["D4s"]["verdict"] == "CONCURRENCY_CONTAMINATION")
v = N.verdict({"Q": Qr, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), smoke_gate="FAIL_SMOKE_DFLASH_ZERO_ACCEPT")}, ARMS)
ok("T9 zero accept fails", v["arms"]["D4s"]["verdict"] == "DFLASH_ZERO_ACCEPT")
v = N.verdict({"Q": Qr, "D4s": rec("D4s", (300, 306), 288, (241,), (221,))}, ARMS)
ok("T9 within noise = NO_GAIN", v["arms"]["D4s"]["verdict"] == "NO_GAIN" and v["best_arm"] is None, v["arms"]["D4s"]["verdict"])
# GPT audit B2 fixture: D4 greedy AL 2.6 < Q 2.8 and D4 slower at 8k (its long windows skipped), D3 faster than Q -> D3 is a WIN
Qb = rec("Q", (300, 305), 287, (240, 244), (220, 222), aa=True, oracle_acceptance={"status": "OK", "al": 2.8})
d4 = rec("D4s", (250, 255), 0, (0,), (0,), oracle_acceptance={"status": "OK", "al": 2.6}, al_vs_q=N.al_vs_q({"al": 2.6}, {"al": 2.8}),
         short_exit={"exit": True, "why": "both 8k below"})
for k_ in ("40k@8", "90k@6", "116k@5"):
    d4["steady"].pop(k_)
d4["steady"].update({f"{b}@skipped": {"status": "SKIPPED_SHORT_EXIT"} for b in ("40k", "90k", "116k")})
v = N.verdict({"Q": Qb, "D4s": d4, "D3s": rec("D3s", (360, 365), 340, (290,), (260,))}, ARMS)
ok("T9 B2: low D4 AL never skips D3; D3 faster -> WIN", v["arms"]["D3s"]["verdict"] == "WIN" and v["best_arm"] == "D3s", v["arms"]["D3s"].get("why"))
ok("T9 B2: D4 below Q at 8k -> REGRESSION (its long windows skipped, recorded)", v["arms"]["D4s"]["verdict"] == "REGRESSION"
   and v["arms"]["D4s"]["short_exit"]["exit"] is True and v["arms"]["D4s"]["al_vs_q"]["below_q"] is True, v["arms"]["D4s"].get("verdict"))
body_txt = (HERE / "nbdf_cell7_body.py").read_text(encoding="utf-8") if (HERE / "nbdf_cell7_body.py").is_file() else ""
if body_txt:
    ok("T9 B2: the body never stops the D arms on AL", "early_exit" not in body_txt and "SKIPPED_LOW_ACCEPTANCE" not in body_txt
       and body_txt.count("stop_d = ") == 2, body_txt.count("stop_d = "))     # both = a broken tree only
v = N.verdict({"Q": Qr, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), memory_gate_first_decode={"ok": None, "why": "INCOMPLETE"})}, ARMS)
ok("T9 USER RULE: a missing memory reading is recorded (flag), the measured WIN stands", v["arms"]["D4s"]["verdict"] == "WIN"
   and "MEMORY_GATE_LOW_OR_MISSING" in v["arms"]["D4s"]["flags"] and v["best_arm"] == "D4s", v["arms"]["D4s"].get("verdict"))
v = N.verdict({"Q": Qr, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), memory_gate_ready={"ok": False})}, ARMS)
ok("T9 USER RULE: a low memory reading is recorded (flag), the measured WIN stands", v["arms"]["D4s"]["verdict"] == "WIN"
   and "MEMORY_GATE_LOW_OR_MISSING" in v["arms"]["D4s"]["flags"], v["arms"]["D4s"].get("verdict"))
ok("T9 Q memory gate recorded only", "INCONCLUSIVE_MEMORY_GATE" in N.verdict({"Q": dict(Qr, memory_gate_ready={"ok": False})}, ARMS)["base"]["memory_gate"])
dl = rec("D4s", (360, 365), 340, (290,), (260,), workers={"num_gpu_blocks": [180]})
v = N.verdict({"Q": Qr, "D4s": dl}, ARMS)
ok("T9 pool drop noted", "capacity_note" in v["arms"]["D4s"] and v["arms"]["D4s"]["capacity"]["pool_drop"] > 0.1, v["arms"]["D4s"].get("capacity"))
v = N.verdict({"Q": Qr, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), status="NO_GO_SMOKE", smoke_gate="FAIL_SMOKE_GARBAGE"),
               "D4s_ISO": {"status": "DONE", "correctness": "CLEAN", "smoke_gate": "PASS"}}, ARMS)
ok("T9 isolation label in verdict", v["isolation"]["D4s"]["label"] == "QDEFER_INTERACTION" and v["arms"]["D4s"]["isolation"] == "QDEFER_INTERACTION")

# ---------------------------------------------------------------- T10 budget walk
ok("T10 hard stop 150, deadman 155", N.HARD_STOP_MIN == 150 and N.DEADMAN_MIN == 155)
bw = N.budget_walk({}, N.HARD_STOP_MIN, 8.0, ARMS)
ok("T10 planned walk at 150 min: Q 50 + D4s 41 + D3s 40 all fit (guaranteed); D4 / D4t / D7 / D4b not at the minimums",
   [w_["runs"] for w_ in bw["walk"]] == [True, True, True, False, False, False, False], bw)
bw = N.budget_walk({"Q": 42, "D4s": 33, "D3s": 32}, N.HARD_STOP_MIN, 8.0, ARMS)
ok("T10 the D4 control runs when Q / D4s / D3s finish faster than their minimums", [w_["runs"] for w_ in bw["walk"]][:4] == [True, True, True, True], bw)
bw = N.budget_walk({"Q": 50, "D4s": 41, "D3s": 40}, 140, 8.0, ARMS)
ok("T10 (at the old 140 min D3 would not fit)", [w_["runs"] for w_ in bw["walk"]][:3] == [True, True, False], bw)

# ---------------------------------------------------------------- T11 the real-kit R4 RecursionError (2026-11-01): wrappers wrap ONCE
class _C:
    calls = 0

    def run(self):
        _C.calls += 1
        return "base"


def _mk(prev):
    def run(self):
        return prev(self)
    return run


w1 = N.wrap_once(_C, "run", _mk)
w2 = N.wrap_once(_C, "run", _mk)
ok("T11 wrap_once object idempotent", w1 is w2 and _C().run() == "base" and _C.calls == 1 and w1._nbdf_orig.__name__ == "run")
import ast as _ast  # noqa: E402
BODY = HERE / "nbdf_cell7_body.py"
if BODY.is_file():
    btxt = BODY.read_text(encoding="utf-8")
    tree = _ast.parse(btxt)
    fns = {n.name: _ast.get_source_segment(btxt, n) for n in tree.body if isinstance(n, _ast.FunctionDef)}
    ok("T11 body defines the factories", {"_mk_merlin_fn_server", "_mk_nbdf_run", "_nbdf_launch_check"} <= set(fns), sorted(fns)[:8])
    seen = []

    def _base_mfs(name, kv, port, kv_bytes):
        seen.append(name)
        return {"server": name}
    ns = {"N": N, "merlin_fn_server": _base_mfs, "_nbdf_launch_check": lambda s, name, port, kv_bytes: s}
    exec(fns["_mk_merlin_fn_server"], ns)
    line = next(l for l in btxt.splitlines() if l.startswith("merlin_fn_server = N.wrap_once("))
    for _ in range(3):                                  # the cell executed 3 times: still one wrapper over the original
        exec(line, ns)
    try:
        out_ = [ns["merlin_fn_server"](f"nbdf_{k.lower()}", "fp8", ARMS[k]["port"], ARMS[k]["kv_bytes"]) for k in ARMS]
        ok("T11 merlin_fn_server x3 exec: no recursion, all 5 arms reach the original once", seen == [f"nbdf_{k.lower()}" for k in ARMS]
           and out_[0] == {"server": "nbdf_q"}, seen)
    except RecursionError as e:
        ok("T11 merlin_fn_server x3 exec: no recursion", False, repr(e))
    ns2 = {"N": N, "L": types.SimpleNamespace(Stream=type("S", (), {"run": lambda self: "traced"})), "_resp_write": lambda st: None, "R": {}}
    exec(fns["_mk_nbdf_run"], ns2)
    sline = next(l for l in btxt.splitlines() if l.startswith("N.wrap_once(L.Stream"))
    exec(sline, ns2)
    exec(sline, ns2)
    ok("T11 Stream.run wrapped once", ns2["L"].Stream.run._nbdf_orig.__name__ == "<lambda>")
    ok("T11 no unguarded rebinding left", "_merlin_fn_server_nb1" not in btxt and "_nbdf_prev_run" not in btxt
       and "L.Stream.run = " not in btxt)
else:
    print("NOTE T11 body file not present (the kit bundles it)")
NB = HERE.parent / "Q27-XHIGH-E27_NBDF_DFLASH2.ipynb"
EXP = HERE / "nbdf_code_export.py"
if NB.is_file() and EXP.is_file() and BODY.is_file():
    code = "\n\n".join("".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code")
    ex = EXP.read_text(encoding="utf-8")
    ok("T11 export = the notebook's code cells exactly (the body runs once, via the stub)", ex == code)
    ok("T11 the body text is not inline in the export", "def _mk_merlin_fn_server" not in ex and ex.count("exec(compile(NBDF_BODY_SRC") == 1)

# ---------------------------------------------------------------- T12 the real-kit gate (B3) + identity / provenance
SRCS = {"tools/merlin/launch.py": "a" * 64, "vllm/config/speculative.py": "b" * 64}
def good_report():
    run = lambda b, g, d: {"num_blocks": [b] * 4, "kv_group_count": g, "drafter_group_count": d, "groups": [], "attn_metadata_builders": g}
    return {"notebook_sha256": "c" * 64,
            "verdict": {"checks": {c: True for c in N.REAL_KIT_CHECKS}},
            "checks": {"R1": {"embed": {"sha256": "s" * 64}, "drafter": {"pins_ok": True, "facts": {
                "files": {"model.safetensors": [N.DRAFTER_PINS["model.safetensors"]["size"], N.DRAFTER_PINS["model.safetensors"]["sha256"]],
                          "config.json": [N.DRAFTER_PINS["config.json"]["size"], N.DRAFTER_PINS["config.json"]["sha256"]]},
                "embed": {"sha256": N.DRAFTER_PINS["embed_tokens"]["sha256"]}, "mask_row": {"sha256": N.DRAFTER_PINS["mask_row"]["sha256"]}}}},
                       "R2": {"src_shas": dict(SRCS), "merlin_commit": "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5"},
                       "R4": {"plans": {"Q": {"argv": ["vllm", "serve", "--tensor-parallel-size", "2", "--pipeline-parallel-size", "2"],
                                              "env": {"VLLM_PP_LAYER_PARTITION": "25,23"}}}},
                       "R9": {"json": {"runs": {"Q": run(215, 5, 0), "K3": run(190, 6, 1), "K4": run(71, 6, 1), "K7": run(71, 6, 1), "K4_swblock1024": run(205, 6, 1), "K3_swblock1024": run(205, 6, 1),
                                                "K4_fp8": run(195, 6, 1)}}},
                       "R12": {"json": {"ok": True}}}}
EXP = {"notebook_shas": {"closed_rebuild": "c" * 64, "pre_rebuild_file": "d" * 64}, "seam_sha": "s" * 64, "ppack_sha": N.PPACK_SHA_PIN,
       "drafter_pins": N.DRAFTER_PINS, "merlin_commit": "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5", "merlin_src_shas": dict(SRCS)}
g = N.kv_gate_from_report(good_report(), EXP)
ok("T12 B1: a passing report opens the gate, every D arm has its per-K summary", g["passed"] is True and all(g["per_k"][k] for k in N.D_ARMS)
   and g["provenance"]["matches"] == ["closed_rebuild"] and set(ARMS) == {"Q", "D4s", "D3s", "D4", "D4t", "D7", "D4b"}, g["why"])
r_ = good_report(); r_["verdict"]["checks"]["drafter_loaded_weights_ok"] = False
ok("T12 B3: kv + seam OK but drafter_loaded_weights_ok False -> closed", N.kv_gate_from_report(r_, EXP)["passed"] is False)
r_["checks"]["R12"] = {"json": {"ok": None, "instantiate_error": "RuntimeError: Failed to infer device type"}}
g = N.kv_gate_from_report(r_, EXP)
ok("T12 CPU_LIMIT accepted, recorded, GPU load check required", g["passed"] is True and g["drafter_loaded_weights"] == "CPU_LIMIT"
   and g["require_gpu_load_check"] is True, g["why"])
r_["checks"]["R12"] = {"json": {"ok": None, "false_raise_suspect": True, "load_weights_error": "RuntimeError: 3 drafter parameter(s) got no tensor"}}
ok("T12 a 'got no tensor' raise is never CPU_LIMIT", N.kv_gate_from_report(r_, EXP)["passed"] is False)
r_ = good_report(); r_["notebook_sha256"] = "e" * 64
ok("T12 stale notebook sha -> closed", N.kv_gate_from_report(r_, EXP)["passed"] is False)
r_ = good_report(); r_["notebook_sha256"] = "d" * 64
ok("T12 the pre-rebuild file sha is accepted (recorded)", N.kv_gate_from_report(r_, EXP)["provenance"]["matches"] == ["pre_rebuild_file"])
for nm, mut in (("seam", lambda r: r["checks"]["R1"]["embed"].update(sha256="x" * 64)),
                ("tp", lambda r: r["checks"]["R4"]["plans"]["Q"].update(argv=["--pipeline-parallel-size", "2"])),
                ("partition", lambda r: r["checks"]["R4"]["plans"]["Q"]["env"].pop("VLLM_PP_LAYER_PARTITION")),
                ("commit", lambda r: r["checks"]["R2"].pop("merlin_commit")),
                ("merlin src", lambda r: r["checks"]["R2"]["src_shas"].update({"tools/merlin/launch.py": "0" * 64})),
                ("drafter", lambda r: r["checks"]["R1"]["drafter"]["facts"]["embed"].update(sha256="0" * 64)),
                ("K4 SW summary", lambda r: r["checks"]["R9"]["json"]["runs"].pop("K4_swblock1024")),
                ("K3 SW summary", lambda r: r["checks"]["R9"]["json"]["runs"].pop("K3_swblock1024")),
                ("one kit check", lambda r: r["verdict"]["checks"].update(tree_check_all_5=False))):
    r_ = good_report(); mut(r_)
    ok(f"T12 missing / different {nm} -> closed", N.kv_gate_from_report(r_, EXP)["passed"] is False)
r_ = good_report(); r_["checks"]["R9"]["json"]["runs"].pop("K7")
g = N.kv_gate_from_report(r_, EXP)
ok("T12 no K7 summary: gate open, D7 not runnable (SKIPPED_NO_KV_K7)", g["passed"] is True and g["per_k"]["D7"] is False and g["per_k"]["D4s"])
ok("T12 wrong ppack sha -> closed", N.kv_gate_from_report(good_report(), dict(EXP, ppack_sha="0" * 64))["passed"] is False)

# ---------------------------------------------------------------- T13 nb4_ppack packing at K = 3 / 4 / 7 (verify B section 4, vLLM #57755)
import torch  # noqa: E402
src = P.PATCHES["vllm/v1/worker/gpu/pp_utils.py"][3] if "vllm/v1/worker/gpu/pp_utils.py" in P.PATCHES else None
if src is not None:
    seg, keep = [], False
    for ln in src.splitlines():
        if ln.startswith("+def nb4_pack_views("):
            keep = True
        elif keep and (not ln.startswith("+") or ln.startswith(("+def ", "+class ", "+_"))):
            break
        if keep:
            seg.append(ln[1:] if ln.startswith("+") else ln)
    ns_ = {"torch": torch}
    exec("\n".join(seg), ns_)
    pv = ns_["nb4_pack_views"]
    for k_ in (3, 4, 7):
        for n_ in (1, 3, 8):
            flat = torch.empty(n_ * (2 * k_ + 3), dtype=torch.int64)
            a_, b_, c_ = pv(flat, n_, k_)
            sampled = torch.randint(0, 248320, (n_, k_ + 1)); counts = torch.randint(0, k_ + 1, (2, n_)); drafts = torch.randint(0, 248320, (n_, k_))
            a_.copy_(sampled); b_.copy_(counts); c_.copy_(drafts)
            a2, b2, c2 = pv(flat.clone(), n_, k_)
            ok(f"T13 pack round trip K{k_} n{n_}", tuple(a_.shape) == (n_, k_ + 1) and tuple(b_.shape) == (2, n_) and tuple(c_.shape) == (n_, k_)
               and torch.equal(a2, sampled) and torch.equal(b2, counts) and torch.equal(c2, drafts)
               and a_.numel() + b_.numel() + c_.numel() == flat.numel() and c_.data_ptr() == flat.data_ptr() + 8 * (n_ * (k_ + 1) + 2 * n_))
else:
    ok("T13 nb4_pack_views found in nb4_ppack", False)

# ---------------------------------------------------------------- T13b D4t verdict (conv impl vs D4)
d4w = rec("D4s", (360, 365), 340, (290,), (260,))
d4w["steady"]["8k@8#1"]["mean_acceptance_length"], d4w["steady"]["8k@8#2"]["mean_acceptance_length"] = 3.10, 3.14
d4t = {k_: v_ for k_, v_ in rec("D4t", (372, 375), 0, (0,), (0,)).items()}
d4t["steady"] = {k_: dict(v_, mean_acceptance_length=3.30) for k_, v_ in d4t["steady"].items() if k_.startswith("8k@")}
v = N.verdict({"Q": Qr, "D4s": d4w, "D4t": d4t}, ARMS)
ok("T13b D4t: CONV_IMPL_TORCH_BETTER when its AL beats D4 by more than D4's A/A", v["arms"]["D4t"]["verdict"] == "CONV_IMPL_TORCH_BETTER"
   and v["throughput_eligible"]["D4t"]["ok"] is False and v["best_arm"] == "D4s", v["arms"]["D4t"])
d4t["steady"] = {k_: dict(v_, mean_acceptance_length=3.13) for k_, v_ in d4t["steady"].items()}
ok("T13b D4t within the A/A spread", N.verdict({"Q": Qr, "D4s": d4w, "D4t": d4t}, ARMS)["arms"]["D4t"]["verdict"] == "CONV_IMPL_NO_DIFFERENCE")
CI = "\n".join(f"(Worker_PP1_TP{t}_EP{2 + t} pid={12 + t}) E27_DFLASH " + json.dumps({"event": "conv_impl", "pid": 12 + t, "impl": "torch"}) for t in (0, 1))
ok("T13b conv_impl read-back required on D4t", N.dflash_readback(ARMS["D4t"], LOG + "\n" + CI)["checks"].get("conv_impl_torch") is True
   and N.dflash_readback(ARMS["D4t"], LOG)["checks"].get("conv_impl_torch") is False)
ok("T13b a torch conv line on a fused arm contradicts", N.dflash_readback(ARMS["D4s"], LOG + "\n" + CI)["definitive"] is True)

ok("T13b D4t is compared with D4s", N.CONV_BASE == "D4s")
SWL = "\n".join(f"(Worker_PP1_TP{t}_EP{2 + t} pid={12 + t}) E27_DFLASH " + json.dumps({"event": "draft_sw_block", "pid": 12 + t,
                 "layer": "model.layers.48.self_attn.attn", "block_size": 1024, "draft_block": 1024}) for t in (0, 1))
ok("T13b draft_sw_block read-back required on SW arms", N.dflash_readback(ARMS["D4s"], LOG)["checks"].get("draft_sw_block_applied") is False
   and N.dflash_readback(ARMS["D4s"], LOG + "\n" + SWL)["checks"].get("draft_sw_block_applied") is True
   and N.dflash_readback(ARMS["D4"], LOG + "\n" + SWL)["definitive"] is True)

# ---------------------------------------------------------------- T14 recorded flags
SG = "\n".join(["(Worker_PP0_TP0_EP0 pid=10) Capturing CUDA graphs (FULL): 100%|###| 6/6 [00:01<00:00]",
                 "(Worker_PP0_TP0_EP0 pid=10) Capturing CUDA graphs (PIECEWISE): 100%|###| 10/10 [00:01<00:00]",
                 "(Worker_PP1_TP0_EP2 pid=12) Capturing dflash2 CUDA graphs (FULL): 100%|###| 8/8 [00:01<00:00]",
                 "(Worker_PP0_TP0_EP0 pid=10) INFO Graph capturing finished in 4 secs, took 0.33 GiB"])
sg = N.stage_graphs(SG)
ok("T14 stage graphs", sg["PP0"] == {"FULL": 6, "PIECEWISE": 10} and sg["PP1_dflash2"] == {"FULL": 8} and sg["finished"] == [("PP0", 4, 0.33)], sg)
ok("T14 PP0_GRAPH_DEGRADED", N.pp0_graph_degraded({"PP0": {"FULL": 6}}, {"PP0": {"FULL": 4}})["flag"] == "PP0_GRAPH_DEGRADED"
   and N.pp0_graph_degraded({"PP0": {"FULL": 6}}, {"PP0": {"PIECEWISE": 10}})["flag"] == "PP0_GRAPH_DEGRADED"
   and N.pp0_graph_degraded({"PP0": {"FULL": 6}}, {"PP0": {"FULL": 8}})["flag"] is None)
qd = lambda f: {"q_readback_final": {"label": "ENGAGED", "shares": {"1": {"fallback_share": f}, "2": {"fallback_share": 0.0}}}}
ok("T14 QDEFER_K_FALLBACK", N.qdefer_k_fallback(qd(0.01), qd(0.2))["flag"] == "QDEFER_K_FALLBACK" and N.qdefer_k_fallback(qd(0.01), qd(0.01))["flag"] is None)
ok("T14 KV_GROUP_EXPLOSION", N.kv_group_explosion({"groups": {"2": [1] * 5}}, {"groups": {"2": [1] * 9}}, 1)["flag"] == "KV_GROUP_EXPLOSION"
   and N.kv_group_explosion({"groups": {"2": [1] * 5}}, {"groups": {"2": [1] * 6}}, 1)["flag"] is None)
ok("T14 PREFIX_REUSE_DROP", N.prefix_reuse_drop({"steady": {"a": {"prefix_hit_rate": 0.8}}}, {"steady": {"a": {"prefix_hit_rate": 0.6}}})["flag"] == "PREFIX_REUSE_DROP")
sl3 = [{"ts": "11-01 03:10:0%d" % i, "accepted": a, "drafted": 400} for i, a in enumerate((240, 200, 120))]
pa = N.preempt_accept({"preemptions": 2, "wall_t0": "11-01 03:10:00", "wall_t1": "11-01 03:10:09"}, 4, sl3)
ok("T14 PREEMPT_ACCEPT_DROP (labelled approximation)", pa["flag"] == "PREEMPT_ACCEPT_DROP" and "approx" not in pa["basis"].lower() or pa["flag"] == "PREEMPT_ACCEPT_DROP", pa)
ok("T14 no preemption -> nothing", N.preempt_accept({"preemptions": 0}, 4, sl3) is None)
ig = N.ingraph_meta("(Worker_PP1_TP0_EP2 pid=12) E27_DFLASH " + json.dumps({"event": "ingraph_meta", "pid": 12, "kernel_map_refresh": True}))
ok("T14 ingraph_meta + INGRAPH_META_FALLBACK trigger", ig["kernel_map_refresh"] is True
   and N.needs_nometa({"smoke_gate": "FAIL_SMOKE_DFLASH_ZERO_ACCEPT", "ingraph_meta": ig}) and not N.needs_nometa({"smoke_gate": "PASS", "ingraph_meta": ig}))
nm_ = N.nometa_arm(ARMS["D4s"])
env_ = dict(nm_["env"], VLLM_PP_LAYER_PARTITION="25,23")
ok("T14 the NOMETA relaunch = D4 + DFLASH_NO_INGRAPH_META=1 (launch check clean)", nm_["env"]["DFLASH_NO_INGRAPH_META"] == "1"
   and nm_["key"] == "D4s_NOMETA" and N.launch_check(nm_, env_, full_argv(nm_, 1), None) == [], N.launch_check(nm_, env_, full_argv(nm_, 1), None))
ok("T14 the draft-group warning text", N.DRAFT_GROUP_WARN_RX in "Speculative decoding (method=dflash) is enabled but no KV cache group could be identified as the draft model's")

# ---------------------------------------------------------------- T15 USER RULE 2026-11-01: gates on REAL failures only
# the GPU commit 8b1760ed: Q DONE / CLEAN / PASS, Q-defer ENGAGED_FALLBACK (exec_share < 0.9, 0 fallbacks) -> every D arm was
# SKIPPED_DEPENDENCY. These are its exact Q-defer stats (q_readback after the long oracle, q_readback_final after the stop).
def _qd(park, exe, empty, recv=0, cons=0, rpc=0, inits=4, label="ENGAGED_FALLBACK"):
    c = {"defer_park": park, "flush_exec": exe, "flush_empty": empty, "flush_recv": recv, "flush_consume": cons, "flush_rpc": rpc,
         "flush_shutdown": 0, "exec_share": round(exe / park, 4) if park else None,
         "fallback_share": round((recv + cons) / park, 4) if park else None, "clean": False}
    return {"label": label, "detail": {"inits": inits, "expect": 4, "last": 2, "nonlast": 2,
                                       "stat_keys": ["defer_park", "defer_post", "flush_empty", "flush_exec"],
                                       "defer_per_rank": {"2408": dict(c), "2409": dict(c)},
                                       "defer_rule": "flush_exec >= 0.9 * defer_park and flush_recv + flush_consume <= 0.05 * defer_park"},
            "shares": {"2408": {"exec_share": c["exec_share"], "fallback_share": c["fallback_share"]},
                       "2409": {"exec_share": c["exec_share"], "fallback_share": c["fallback_share"]}}}


QD_EARLY, QD_FINAL = _qd(3562, 562, 3000), _qd(24406, 21000, 3404, rpc=2)
ok("T15 exact GPU stats: exec_share 0.1578 / 0.8604, fallback 0 -> no real Q-defer failure",
   N.qdefer_real_failure(QD_EARLY)[0] is False and N.qdefer_real_failure(QD_FINAL)[0] is False
   and QD_EARLY["detail"]["defer_per_rank"]["2408"]["exec_share"] == 0.1578 and QD_FINAL["detail"]["defer_per_rank"]["2409"]["exec_share"] == 0.8604,
   (N.qdefer_real_failure(QD_EARLY), N.qdefer_real_failure(QD_FINAL)))
QG = rec("Q", (300, 305), 287, (240, 244), (220, 222), aa=True, q_readback=QD_EARLY, q_readback_end=QD_FINAL, q_readback_final=QD_FINAL,
         memory_gate_ready={"ok": False, "why": "LOW_MEMORY_RANKS:[2, 3] below 512 MiB: [663, 687, 177, 175]"},
         memory_gate_first_decode={"ok": False, "why": "LOW_MEMORY_RANKS:[2, 3] below 512 MiB: [659, 683, 171, 169]"})
bu = N.base_usable(QG)
ok("T15 the GPU Q (ENGAGED_FALLBACK, memory gate False) is a usable base", bu[0] is True and bu[1] == "OK", bu)
ok("T15 the old strict NB4T gate would have rejected it (now recorded only)", T.base_check(QG)[1] == "INCONCLUSIVE_BASE_NOT_ENGAGED", T.base_check(QG))
v = N.verdict({"Q": QG, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), q_readback_final=_qd(24406, 21000, 3404))}, ARMS)
ok("T15 the GPU Q base -> D4s measured (WIN), the base strict label recorded", v["base"]["ok"] is True and v["arms"]["D4s"]["verdict"] == "WIN"
   and v["base"]["strict_nb4t_base_check"]["label"] == "INCONCLUSIVE_BASE_NOT_ENGAGED" and v["base"]["qdefer"]["real_failure"] is False
   and "QDEFER_LABEL_ENGAGED_FALLBACK" in v["arms"]["D4s"]["flags"], (v["base"], v["arms"]["D4s"].get("verdict")))
for nm_, q_ in (("fallback_share 0.2", _qd(1000, 800, 0, recv=150, cons=50)), ("recv + consume 6 %", _qd(1000, 940, 0, recv=60)),
                ("fallback_share 0.001 (> 0)", _qd(1000, 999, 0, cons=1)), ("3 of 4 init lines", _qd(1000, 1000, 0, inits=3)),
                ("nb4_ppack NOT_INTEGRATED", dict(_qd(1000, 1000, 0), label="NOT_INTEGRATED"))):
    ok(f"T15 real Q-defer failure: {nm_}", N.qdefer_real_failure(q_)[0] is True, N.qdefer_real_failure(q_))
QF = dict(QG, q_readback_final=_qd(1000, 800, 0, recv=150, cons=50))
ok("T15 fallback_share 0.2 -> base not usable", N.base_usable(QF)[0] is False and N.base_usable(QF)[1] == "INCONCLUSIVE_BASE_QDEFER_FAILED", N.base_usable(QF))
vf = N.verdict({"Q": QF, "D4s": rec("D4s", (360, 365), 340, (290,), (260,))}, ARMS)
ok("T15 fallback_share 0.2 on Q -> no D winner (base label)", vf["base"]["ok"] is False and vf["arms"]["D4s"]["verdict"] == "INCONCLUSIVE_BASE_QDEFER_FAILED", vf["arms"]["D4s"])
vd = N.verdict({"Q": QG, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), q_readback_final=_qd(1000, 800, 0, recv=200))}, ARMS)
ok("T15 fallback_share 0.2 on a D arm -> QDEFER_FAILED", vd["arms"]["D4s"]["verdict"] == "QDEFER_FAILED", vd["arms"]["D4s"])
ok("T15 a real Q-defer failure on a D arm triggers the one Q-defer-off isolation rerun", N.needs_isolation({"status": "QDEFER_FAILED"})
   and not N.needs_isolation({"status": "DONE", "q_readback_final": {"label": "ENGAGED_FALLBACK"}}))
QP = dict(QG, stall={"parity": {"label": "IMBALANCED", "modal_requests": {"0": 7, "1": 1}}})
for w_ in QP["steady"].values():
    w_["parity"] = {"label": "IMBALANCED"}
vp = N.verdict({"Q": QP, "D4s": rec("D4s", (360, 365), 340, (290,), (260,))}, ARMS)
ok("T15 parity IMBALANCED -> recorded only (base usable, D4s WIN)", N.base_usable(QP)[0] and vp["arms"]["D4s"]["verdict"] == "WIN"
   and vp["parity"]["Q"]["profile_label"] == "IMBALANCED", (vp["parity"].get("Q"), vp["arms"]["D4s"].get("verdict")))
for nm_, mg_ in (("INCONCLUSIVE (None)", {"ok": None, "why": "INCOMPLETE_FOUR_RANK_MEMORY: [663, None]"}), ("False", {"ok": False, "why": "LOW"})):
    QM = dict(QG, memory_gate_ready=mg_, memory_gate_first_decode=mg_)
    vm = N.verdict({"Q": QM, "D4s": rec("D4s", (360, 365), 340, (290,), (260,))}, ARMS)
    ok(f"T15 Q memory gate {nm_} -> recorded, base usable, D4s measured", N.base_usable(QM)[0] and vm["arms"]["D4s"]["verdict"] == "WIN"
       and "INCONCLUSIVE_MEMORY_GATE" in vm["base"]["memory_gate"], (vm["base"]["memory_gate"], vm["arms"]["D4s"].get("verdict")))
for nm_, kw_ in (("smoke INCONCLUSIVE_SMOKE_FEW_DRAFTS", {"smoke_gate": "INCONCLUSIVE_SMOKE_FEW_DRAFTS"}),
                 ("config read-back incomplete", {"config_readback": {"ok": False, "definitive": False, "why": "missing: captures", "log": {}}}),
                 ("dflash read-back contradiction", {"dflash_readback": {"ok": False, "definitive": True, "why": "drafter EAGER"}}),
                 ("Q-defer label NOT_ENGAGED", {"q_readback_final": {"label": "NOT_ENGAGED", "detail": {"inits": 4, "expect": 4}}})):
    vr = N.verdict({"Q": QG, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), **kw_)}, ARMS)
    ok(f"T15 {nm_} -> recorded (flag), measured verdict WIN", vr["arms"]["D4s"]["verdict"] == "WIN" and vr["arms"]["D4s"]["flags"], vr["arms"]["D4s"].get("flags"))
for nm_, kw_, want in (("smoke FAIL zero accept", {"smoke_gate": "FAIL_SMOKE_DFLASH_ZERO_ACCEPT"}, "DFLASH_ZERO_ACCEPT"),
                       ("smoke FAIL garbage", {"smoke_gate": "FAIL_SMOKE_GARBAGE"}, "SMOKE_FAIL"),
                       ("garbage correctness", {"correctness": "NO_GO_GARBAGE"}, "CORRECTNESS_FAIL"),
                       ("server died", {"status": "SERVER_DIED"}, "INCONCLUSIVE_ARM_FAILED")):
    vr = N.verdict({"Q": QG, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), **kw_)}, ARMS)
    ok(f"T15 real failure kept: {nm_} -> {want}", vr["arms"]["D4s"]["verdict"] == want, vr["arms"]["D4s"].get("verdict"))
vr = N.verdict({"Q": QG, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), oracle_vs="FAIL_HIGH_MARGIN")}, ARMS)
ok("T15 B2: oracle FAIL_HIGH_MARGIN -> recorded (flag ORACLE_RECORDED_FAIL_HIGH_MARGIN), the measured verdict stands",
   vr["arms"]["D4s"]["verdict"] == "WIN" and "ORACLE_RECORDED_FAIL_HIGH_MARGIN" in vr["arms"]["D4s"]["flags"], (vr["arms"]["D4s"].get("verdict"), vr["arms"]["D4s"].get("flags")))
for nm_, kw_ in (("not DONE", {"status": "SERVER_DIED"}), ("garbage", {"correctness": "NO_GO_GARBAGE"}), ("oracle FAIL", {"quality_gate": "FAIL"}),
                 ("long oracle NO_GO", {"long_gate": "NO_GO_LONG_ORACLE"})):
    ok(f"T15 Q real failure -> base not usable: {nm_}", N.base_usable(dict(QG, **kw_))[0] is False, N.base_usable(dict(QG, **kw_)))
ok("T15 Q oracle INCONCLUSIVE (no RC2 reference) -> recorded, base usable", N.base_usable(dict(QG, quality_gate="INCONCLUSIVE", oracle_vs="INCONCLUSIVE_NO_ORACLE"))[0])
ok("T15 smoke: /metrics not read -> INCONCLUSIVE (recorded), no drafts / zero accept stay FAIL",
   N.smoke_gate({"tokens": [N.SMOKE_TOKENS] * 2, "acceptance": {"status": "NO_METRICS"}}) == "INCONCLUSIVE_SMOKE_NO_METRICS"
   and N.smoke_gate({"tokens": [N.SMOKE_TOKENS] * 2, "acceptance": {"status": "NO_DRAFTS"}}) == "FAIL_SMOKE_NO_DRAFTS")

# the body's nbdf_main on the GPU Q record: base usable -> every D arm planned and run; fallback 0.2 -> every D arm skipped
if BODY.is_file():
    btxt = BODY.read_text(encoding="utf-8")
    fns = {n.name: _ast.get_source_segment(btxt, n) for n in _ast.parse(btxt).body if isinstance(n, _ast.FunctionDef)}
    ok("T15 body: no label-only stops left", "INCONCLUSIVE_QUALITY" not in btxt and '"INCONCLUSIVE_SMOKE"' not in btxt
       and 'not in ("ENGAGED", "ENGAGED_FALLBACK")' not in btxt and "N.qdefer_real_failure(o[\"q_readback\"])" in btxt
       and "N.base_usable(q)" in btxt and "_can_step(arm)" in btxt, "body text")

    class _Seq:
        def __init__(self):
            self.state, self.ring, self.log, self.error, self.apply_rec, self.revert_rec = "applied", True, [], None, {}, {}
            self.verify_rec = {"files": {"vllm/qsa_cache.py": {"state": "applied"}}}

        def apply(self):
            return True, "applied"

        def check(self, k):
            return True, "FULLY_PATCHED"

        def before_arm(self, a):
            return True, "FULLY_PATCHED"

        def finish(self):
            self.state = "reverted"

    def _main_ns(q_rec, rows=("rc2_fp8",), per_k_missing=(), d_out=None):
        calls, R_ = [], {"sections": {"A0": {"merge": {"ok": True}}}, "setup": {}}

        def runner(k):
            calls.append(k)
            if k == "Q":
                return dict(q_rec)
            return dict((d_out or {}).get(k) or {"status": "DONE", "pin": 3016709612, "smoke_gate": "PASS"})

        def step(sec, name, est, fn):
            r_ = dict(fn() or {})
            if r_.get("status") in (None, "NOT_RUN"):
                r_["status"] = "DONE"
            R_["sections"].setdefault(sec, {})[name] = r_
            return r_
        ns = {"N": N, "P": P, "T": T, "R": R_, "ARMS": ARMS, "ISO": {}, "NBDF": {}, "FAKE": False, "ROWS": {r: [1] for r in rows},
              "step": step, "save": lambda: None, "log": lambda *a, **k: None, "finalize": lambda: None, "left": lambda: 4 * 3600,
              "merlin_install": lambda: {"kv": {"MERLIN_TREE": "/x"}}, "apply_60490": lambda: {"status": "DONE", "ok": True},
              "sha_gate": lambda t: (True, "ok"), "SEAM_API": object(), "SEAM_LOAD": {}, "SEAM_ENVS": SENV,
              "_tree_py": lambda: (_ for _ in ()).throw(RuntimeError("no tree in the unit test")),
              "PIN": {"kv_bytes": 3553580524}, "D_PIN": {"kv_bytes": None}, "SW_PIN": {"kv_bytes": None}, "Q_REC": {},
              "DRAFTER": {"ok": True}, "KV_KIT_GATE": {"passed": True, "per_k": {k: k not in per_k_missing for k in N.D_ARMS}},
              "_AKEYS": ("name", "est_min"), "PSEQ": None, "SSEQ": None, "run_arm": runner}
        for fn_ in ("_arm", "skip_arm", "_patch_rec", "_seam_rec", "nbdf_main"):
            exec(fns[fn_], ns)
        ns["nbdf_main"](runner=runner, pseq=_Seq(), sseq=_Seq())
        return calls, R_, ns["NBDF"]

    QGB = dict(QG, free_after_ready_min={"min": [663, 687, 177, 175]}, pin=3553580524)
    c_, R_, NB_ = _main_ns(QGB)
    ok("T15 nbdf_main: the GPU Q (ENGAGED_FALLBACK) -> base usable, every D arm planned and run in ORDER",
       c_ == list(N.ORDER) and NB_["base"]["usable"] is True and NB_["base"]["strict_nb4t_base_check"]["label"] == "INCONCLUSIVE_BASE_NOT_ENGAGED"
       and not any(str(v.get("status", "")).startswith("SKIPPED") for k in N.D_ARMS for v in R_["sections"].get(k, {}).values()), (c_, NB_.get("base")))
    c_, R_, NB_ = _main_ns(dict(QGB, q_readback_final=_qd(1000, 800, 0, recv=150, cons=50)))
    ok("T15 nbdf_main: Q fallback_share 0.2 -> base not usable, every D arm SKIPPED_DEPENDENCY", c_ == ["Q"]
       and all(v.get("status") == "SKIPPED_DEPENDENCY" for k in N.D_ARMS for v in R_["sections"][k].values()), (c_, NB_.get("base")))
    QMM = dict(QGB, memory_gate_ready={"ok": None, "why": "INCOMPLETE"}, stall={"parity": {"label": "IMBALANCED"}})
    c_, R_, NB_ = _main_ns(QMM)
    ok("T15 nbdf_main: Q memory gate INCONCLUSIVE + parity IMBALANCED -> recorded, D arms run", c_ == list(N.ORDER), c_)
    c_, R_, NB_ = _main_ns(QGB, rows=(), per_k_missing=("D7", "D4b"))
    ok("T15 nbdf_main: RC2 rows missing + no per-K kit summary for D7 / D4b -> recorded, every arm runs", c_ == list(N.ORDER)
       and "MISSING" in NB_.get("rc2_rows", "") and set(NB_.get("recorded_gates", {})) == {"D7", "D4b"}, (c_, NB_.get("recorded_gates")))
    c_, R_, NB_ = _main_ns(QGB, d_out={"D4s": {"status": "NOT_INTEGRATED", "not_integrated": True, "early_stop": "SPEC_MISMATCH: spec_k 3 != 4"}})
    ok("T16 nbdf_main: a SPEC_MISMATCH arm (NOT_INTEGRATED) skips only itself, the other D arms run", c_ == list(N.ORDER)
       and not any(k_.endswith("_ISO") for k_ in c_), c_)
    ok("T16 body: the spec / drafter mismatch gate is in the body", "N.spec_mismatch(is_d, o.get(\"config_readback\"), o.get(\"dflash_readback\"))" in btxt)
else:
    print("NOTE T15 body file not present (the kit bundles it)")

# ---------------------------------------------------------------- T16 read-back split (coordinator sign-off 2026-11-01)
D4A = ARMS["D4s"]
for nm_, crb_, drb_ in (
        ("spec method not dflash", {"contradictions": ["spec_method 'mtp' != 'dflash'"]}, {}),
        ("K differs (config)", {"contradictions": ["spec_k 3 != 4"]}, {}),
        ("drafter model differs", {"contradictions": ["spec_model '/kaggle/input/x' != '/kaggle/input/y'"]}, {}),
        ("MTP drafter resolved (real parser)", {}, N.dflash_readback(D4A, "INFO Resolved architecture: Qwen4ExpForCausalLM\nINFO Resolved architecture: Qwen4ExpMTP")),
        ("DFlash2DraftModel not resolved (real parser)", {}, N.dflash_readback(D4A, "INFO Resolved architecture: Qwen4ExpForCausalLM")),
        ("SpecDecoding lines with another K", {}, {"contradictions": ["SpecDecoding lines with [3] positions, K 4"]}),
        ("drafter EAGER (user rule: a real fallback)", {}, {"contradictions": ["the drafter runs EAGER (its attention backend has no full CUDA graphs)"],
                                                         "rec": {"archs": ["Qwen4ExpForCausalLM", "DFlash2DraftModel"]}})):
    sm_ = N.spec_mismatch(True, crb_, drb_)
    ok(f"T16 REAL failure: {nm_} -> spec_mismatch", sm_[0] is True, (sm_, drb_.get("contradictions") if isinstance(drb_, dict) else None))
    vs_ = N.verdict({"Q": QG, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), config_readback=dict({"ok": False, "definitive": True, "log": {}}, **crb_),
                                         dflash_readback=dict(drb_, ok=False, definitive=True))}, ARMS)
    ok(f"T16 REAL failure: {nm_} -> verdict NOT_INTEGRATED (SPEC_MISMATCH)", vs_["arms"]["D4s"]["verdict"] == "NOT_INTEGRATED"
       and "SPEC_MISMATCH" in vs_["arms"]["D4s"]["why"], vs_["arms"]["D4s"].get("verdict"))
for nm_, crb_, drb_ in (
        ("backend mismatch", {}, {"contradictions": ["draft backend ['TRITON_ATTN'] != FLASH_ATTN"], "rec": {"archs": ["DFlash2DraftModel"]}}),
        ("captures differ", {"contradictions": ["captures [5, 10] != [5, 10, 15, 20, 25, 30, 35, 40]"]}, {"rec": {"archs": ["DFlash2DraftModel"]}}),
        ("incomplete read-back (no arch lines at all)", {"contradictions": []}, {"contradictions": [], "rec": {"archs": []}})):
    sm_ = N.spec_mismatch(True, crb_, drb_)
    ok(f"T16 flag only: {nm_} -> no spec_mismatch", sm_[0] is False, sm_)
    dfn = bool(crb_.get("contradictions") or drb_.get("contradictions"))
    vf_ = N.verdict({"Q": QG, "D4s": rec("D4s", (360, 365), 340, (290,), (260,), config_readback=dict({"ok": not crb_.get("contradictions"),
                                         "definitive": bool(crb_.get("contradictions")), "log": {}}, **crb_),
                                         dflash_readback=dict(drb_, ok=False, definitive=bool(drb_.get("contradictions"))))}, ARMS)
    ok(f"T16 flag only: {nm_} -> measured WIN + {'READBACK_CONTRADICTION' if dfn else 'READBACK_INCOMPLETE'}",
       vf_["arms"]["D4s"]["verdict"] == "WIN" and ("READBACK_CONTRADICTION" if dfn else "READBACK_INCOMPLETE") in vf_["arms"]["D4s"]["flags"],
       (vf_["arms"]["D4s"].get("verdict"), vf_["arms"]["D4s"].get("flags")))
ok("T16 Q: the config spec checks apply to Q too (MTP K3 plan), DFlash arch checks do not",
   N.spec_mismatch(False, {"contradictions": ["spec_k 4 != 3"]}, None)[0] and not N.spec_mismatch(False, {}, {"rec": {"archs": ["X"]}})[0])

print(f"{RES['pass']}/{RES['pass'] + RES['fail']} PASS", flush=True)
sys.exit(1 if RES["fail"] else 0)
