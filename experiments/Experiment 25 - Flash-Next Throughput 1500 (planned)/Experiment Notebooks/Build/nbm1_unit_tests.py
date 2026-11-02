"""E25 NB-M1 quick unit tests (local + the Py3.12 CPU kit), on the real nbdf_lib / nb4t_lib / nb4_ppack and the BUILT body:
  M1  the arm plan: Q = NBDF-K100's Q (MTP K3, seqs 24, pin 7,933,526,016, Q-defer); every lever = Q + exactly one change
      (check_arms); each change's env / argv / spec / captures / seqs / head; the window sets (90k + 116k always)
  M2  install(): nbdf_lib's arm-set constants rebound (ORDER, D_ARMS = the levers, no SW / short arm, K_OF)
  M3  launch_check: every arm's planned launch passes (after apply_head with the int8 head and with the BF16 head); a
      lever's change on another arm / a wrong head / a wrong K is refused
  M4  the read-backs (real failures): each change present on its arm (env in 4 workers, REDUCED DRAFT VOCAB lines, head
      lines, indexer dtype, logged seqs / batched tokens / long-prefill / spec K / schedule) and absent elsewhere; PROF
  M5  the switches + the three copies (Q in each, every lever in exactly one copy, PROF in C) + the time plan (every copy's
      walk runs all its arms, runtime <= COPY_RUNTIME_MAX, BUDGET_MIN = the max walk + 5 min rounded up)
  M6  the built body: nbdf_main (fakes) launches Q then the levers; the copy switches; Q off; a failed lever never skips
      the next; the closed kit gate; a missing vocab table; the NB-M1 edits present; the switch line read from the code
      EXPORT; arm_boot (fakes): an OOM at any stage (first request after ready) steps the Q pin ladder, an Xid does not;
      the lever head into Merlin's build env
  M7  nbdf_lib.verdict over the NB-M1 arms (Q = the base) + the records
Run: python nbm1_unit_tests.py   -> 'FAIL <name>' lines, then '<pass>/<total> PASS'."""
import ast
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
for d in (HERE, E27B, HERE / "e27"):
    if d.is_dir():
        sys.path.insert(0, str(d))
import nbdf_lib as N  # noqa: E402
import nb4t_lib as T  # noqa: E402
import nb4_ppack as P  # noqa: E402
import nbm1_lib as M1  # noqa: E402

RES = {"pass": 0, "fail": 0}


def ok(name, cond, info=""):
    RES["pass" if cond else "fail"] += 1
    if not cond:
        print("FAIL", name, str(info)[:400], flush=True)


QCAPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96]
N.configure(SEQS=24, CONC_LADDER=(2, 8, 23), Q_CAPTURES=QCAPS, Q_TAIL=["--compilation-config", N.compilation_json(QCAPS)],
            WKEYS=("8k@16#1", "8k@16#2", "profile_8k@16", "40k@16"), PREFIX_RECORD_ONLY=True, SHORT_EXIT_ARMS=(),
            LONG_THR_FLOOR={"90k": 0.037, "116k": 0.054}, TF_REQUIRED=True)
Q_EXTRA = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}", "--max-num-seqs", "24"]
PIN = 7933526016
ALL = N.arm_plan(Q_EXTRA, {"VLLM_SERVER_DEV_MODE": "1", "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, PIN,
                 "/kaggle/input/models/foysalemonshanto/" + N.DRAFTER_REL, q_env=P.server_env("Q-defer"))
N.restrict_arms(["Q", "D3s", "D4s"])
K100 = {k: ALL[k] for k in N.ORDER}
Q0 = copy.deepcopy(K100["Q"])
VPATH = {k: f"/kaggle/input/datasets/{M1.VOCAB_DS}/{v[0]}" for k, v in M1.VOCAB_TABLES.items()}
VOCAB = {k: {"path": VPATH[k], "found": True} for k in M1.VOCAB_TABLES}


def argv_of(a):
    return ["serve", "x"] + a["extra"] + ["--kv-cache-memory-bytes", str(PIN)]


def env_of(a, **add):
    e = {k: v for k, v in a["env"].items() if v is not None}
    e.update(VLLM_PP_LAYER_PARTITION="25,23", **add)
    return e


def with_head(a, head):
    """The Merlin argv as build() emits it: its own --speculative-config (+ draft_lm_head_quant when DRAFT_HEAD is set) first."""
    sp = '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"' + (f',"draft_lm_head_quant":"{head}"' if head else "") + "}"
    return ["serve", "x", "--speculative-config", sp, "--max-num-seqs", "8"] + a["extra"] + ["--kv-cache-memory-bytes", str(PIN)]


# ---------------------------------------------------------------- M1 the arm plan
ARMS = M1.m1_arms(N, K100["Q"], None, VOCAB)
ok("M1 arms in order Q V65 V32 I4 IDX S28 S32 LP G1 G2 K4 DYN", tuple(ARMS) == M1.ARM_KEYS, tuple(ARMS))
Q = ARMS["Q"]
ok("M1 Q = NBDF-K100's Q (env, argv, spec, captures, K, pin, port, Q-defer) named nbm1_q, A/A on",
   all(Q[f] == Q0[f] for f in ("env", "extra", "spec", "captures", "k", "kv_bytes", "port", "q_label", "expect_cfg"))
   and Q["name"] == "nbm1_q" and Q["long_aa"] is True and Q["method"] == "mtp" and Q["k"] == 3 and Q["env"].get("NB4_PPDEFER") == "1")
ok("M1 Q: MTP K3 probabilistic, captures capture_sizes_for(24, 3), seqs 24, no trailing spec",
   Q["spec"] == {"method": "mtp", "num_speculative_tokens": 3, "draft_sample_method": "probabilistic"} and Q["captures"] == M1.capture_sizes_for(24, 3)
   and T.last_value(Q["extra"], "--max-num-seqs") == "24" and "--speculative-config" not in Q["extra"])
ok("M1 Q's expect: every M1 env key absent", all(Q["expect"].get(k) is None for k in M1.M1_ENV_KEYS) and set(Q["expect"]) >= set(M1.M1_ENV_KEYS))
ok("M1 the K100 Q arm is untouched (deep copies)", K100["Q"] == Q0)
try:
    M1.check_arms(N, T, ARMS, Q0)
    ca = True
except ValueError as e:
    ca = repr(e)
ok("M1 check_arms: Q = K100 Q, every lever = Q + exactly its change", ca is True, ca)
ok("M1 Q est / cap include PROF (all-on plan) and the time model", Q["est_min"] == M1.q_time(True)[0] and Q["cap_min"] == M1.q_time(True)[1])
for k, (name, sh, n) in M1.VOCAB_TABLES.items():
    a = ARMS[k]
    ok(f"M1 {k}: VLLM_DRAFT_VOCAB_FILE = the found table + VLLM_DRAFT_VOCAB_SHA256 pinned, nothing else",
       a["env"][M1.ENV_VOCAB] == VPATH[k] and a["env"][M1.ENV_VOCAB_SHA] == sh and a["extra"] == Q["extra"] and a["m1_head"] == "q"
       and a["m1_vocab"]["n"] == n and n % (128 * 2) == 0, a["env"])
ok("M1 env names = Merlin draft_head.py ENV_DRAFT_VOCAB_FILE / ENV_DRAFT_VOCAB_SHA256", (M1.ENV_VOCAB, M1.ENV_VOCAB_SHA) == ("VLLM_DRAFT_VOCAB_FILE", "VLLM_DRAFT_VOCAB_SHA256"))
ok("M1 I4: head int4, env / argv = Q", ARMS["I4"]["m1_change"] == {"env": {}, "argv": [], "head": "int4"} and ARMS["I4"]["extra"] == Q["extra"]
   and ARMS["I4"]["env"] == Q["env"])
ok("M1 IDX: + --attention-config JSON (not the dotted form)", ARMS["IDX"]["extra"] == Q["extra"] + ["--attention-config", '{"indexer_kv_dtype":"fp8"}']
   and not any(".indexer_kv_dtype" in x for x in ARMS["IDX"]["extra"]))
ok("M1 LP: + the three long-prefill args", ARMS["LP"]["extra"] == Q["extra"] + ["--max-num-batched-tokens", "2048", "--long-prefill-token-threshold", "1024",
                                                                                 "--long-prefill-token-threshold-adaptive"])
for k, s in M1.SEQ_ARMS.items():
    a = ARMS[k]
    caps = M1.capture_sizes_for(s, 3)
    ok(f"M1 {k}: --max-num-seqs {s} (last wins), captures capture_sizes_for({s}, 3) top {s * 4}, streams {s}, coverage",
       T.last_value(a["extra"], "--max-num-seqs") == str(s) and N.captures_from_argv(a["extra"]) == caps == a["captures"] and caps[-1] == s * 4
       and a["streams"] == s and a["expect_cfg"]["max_num_seqs"] == s and a["coverage"]["covers_all"] and len(a["extra"]) == len(Q["extra"]), caps)
k4, dy = ARMS["K4"], ARMS["DYN"]
ok("M1 K4: spec K 4, trailing spec with Q's int8 head, captures every multiple of 5 up to 120, exact coverage",
   k4["spec"] == {"method": "mtp", "num_speculative_tokens": 4, "draft_sample_method": "probabilistic"} and k4["k"] == 4
   and T.spec_from_argv(k4["extra"]) == dict(k4["spec"], draft_lm_head_quant="int8") and k4["captures"] == [5 * b for b in range(1, 25)]
   and N.captures_from_argv(k4["extra"]) == k4["captures"] and k4["coverage"]["covers_all"] and not k4["coverage"]["padded_b"]
   and k4["expect_cfg"]["num_speculative_tokens"] == 4 and not k4["ring_needed"], k4["captures"])
ok("M1 DYN: K 4 + [[1,8,4],[9,24,3]], captures b x 5 (b <= 8) and b x 4 (9-24), tier coverage",
   dy["spec"]["num_speculative_tokens_per_batch_size"] == [[1, 8, 4], [9, 24, 3]] and dy["k"] == 4
   and all(b * 5 in dy["captures"] for b in range(1, 9)) and all(b * 4 in dy["captures"] for b in range(9, 25))
   and dy["coverage"]["covers_all"] and M1.dense_schedule(M1.DYN_SCHEDULE, 24, 4)[1:] == [4] * 8 + [3] * 16, dy["captures"])
ok("M1 G1 / G2: one GDN switch each", ARMS["G1"]["m1_change"]["env"] == {"VLLM_GDN_DECODE_FASTPATH": "1"}
   and ARMS["G2"]["m1_change"]["env"] == {"VLLM_GDN_SHARED_ALIGN_INDEX": "1"} and ARMS["G1"]["extra"] == Q["extra"])
ok("M1 every lever: A/A-free window set with 90k@8 and 116k@6, no band twice; 8k@16#1 (Q's key) except S28 / S32 (20k at their seqs)",
   all(any(w[:2] == ["90k", 8] for w in a["windows_fixed"]) and any(w[:2] == ["116k", 6] for w in a["windows_fixed"])
       and len({w[0] for w in a["windows_fixed"]}) == len(a["windows_fixed"])
       and (["20k", M1.SEQ_ARMS[k], ""] in a["windows_fixed"] and not any(w[0] == "8k" for w in a["windows_fixed"]) if k in M1.SEQ_ARMS
            else ["8k", 16, "#1"] in a["windows_fixed"])
       for k, a in ARMS.items() if k != "Q"))
ok("M6-pre the K100 Q window list is Q's (8k@16 x 2, 20k@24, 40k@16, 90k@8, 116k@6) and no lever has the A/A pair",
   M1.Q_WINDOWS == [["8k", 16, "#1"], ["8k", 16, "#2"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]]
   and not any(["8k", 16, "#2"] in a["windows_fixed"] for k, a in ARMS.items() if k != "Q"))
ok("M1 levers: the record-only Q items off (ladder, prefill probe, long oracle, prefix, profiles); TF only on IDX / LP",
   all(not M1.item_on(a, x) for a in list(ARMS.values())[1:] for x in ("ladder", "prefill_probe", "long_oracle", "prefix", "profiles"))
   and {k for k, a in ARMS.items() if k != "Q" and M1.item_on(a, "tf")} == {"IDX", "LP"} and all(M1.item_on(Q, x) for x in ("tf", "prefix", "profiles")))
ok("M1 ports distinct, inside Merlin's 8000..8355 with retry / iso offsets", len({a["port"] for a in ARMS.values()}) == len(ARMS)
   and max(a["port"] for a in ARMS.values()) + 8 <= 8355)
for nm_, mut in (("a second change (GDN on V65)", lambda x: x["V65"]["env"].update(VLLM_GDN_DECODE_FASTPATH="1")),
                 ("a lever without 116k", lambda x: x["G1"].update(windows_fixed=[["8k", 16, "#1"], ["90k", 8, ""]])),
                 ("an argv change beyond the plan", lambda x: x["IDX"]["extra"].append("--enforce-eager")),
                 ("a K change on S28", lambda x: x["S28"].update(k=4)),
                 ("Q not K100's", lambda x: x["Q"]["env"].update(NB4_PPDEFER="0"))):
    bad_arms = copy.deepcopy(ARMS)
    mut(bad_arms)
    try:
        M1.check_arms(N, T, bad_arms, Q0)
        r_ = "no raise"
    except ValueError as e:
        r_ = str(e)
    ok(f"M1 check_arms refuses {nm_}", r_ != "no raise", r_)
NOV = M1.m1_arms(N, K100["Q"], None, {})
ok("M1 a vocab table not found -> the arm's pre_gate NOT_INTEGRATED (other arms launch)",
   M1.pre_gate(NOV["V65"], {"passed": True})[0] == "NOT_INTEGRATED" and M1.pre_gate(NOV["G1"], {"passed": True}) is None
   and "MISSING:" in NOV["V32"]["env"][M1.ENV_VOCAB])
ok("M1 the closed kit gate -> every lever SKIPPED_KV_KIT_GATE", all(M1.pre_gate(a, {"passed": False, "why": ["closed"]})[0] == "SKIPPED_KV_KIT_GATE"
                                                                     for k, a in ARMS.items() if k != "Q"))
import hashlib as _hl  # noqa: E402
_files = {f"/in/datasets/{M1.VOCAB_DS}/draft_vocab_k100_arc_n65536.pt": b"x65", f"/in/{M1.VOCAB_DS.split('/')[1]}/draft_vocab_k100_arc_n32768.pt": b"x32"}


def _rd(p):
    p = p.replace("\\", "/")
    if p not in _files:
        raise FileNotFoundError(p)
    return _files[p]


_orig = dict(M1.VOCAB_TABLES)
M1.VOCAB_TABLES = {"V65": ("draft_vocab_k100_arc_n65536.pt", _hl.sha256(b"x65").hexdigest(), 65536),
                   "V32": ("draft_vocab_k100_arc_n32768.pt", _hl.sha256(b"x32").hexdigest(), 32768)}
rv = M1.resolve_vocab("/in", {}, reader=_rd)
M1.VOCAB_TABLES = {"V65": ("draft_vocab_k100_arc_n65536.pt", _hl.sha256(b"other").hexdigest(), 65536), "V32": _orig["V32"]}
rv2 = M1.resolve_vocab("/in", {}, reader=_rd)
M1.VOCAB_TABLES = _orig
ok("M1 resolve_vocab: datasets/<owner>/<slug> and the legacy /<slug> mount, sha-pinned; a wrong sha -> not found",
   rv["V65"]["found"] and rv["V32"]["found"] and rv["V65"]["path"].replace("\\", "/").endswith("datasets/foysalemonshanto/q27-e29-final-payload/draft_vocab_k100_arc_n65536.pt")
   and not rv2["V65"]["found"] and "VOCAB_TABLE_MISSING" in rv2["V65"]["why"], (rv, rv2["V65"]))

# ---------------------------------------------------------------- M2 install
M1.install(N, ARMS)
ok("M2 install: ORDER / D_ARMS = levers / no SW-short-probe arm / K_OF", N.ORDER == M1.ARM_KEYS and N.D_ARMS == M1.LEVERS
   and N.SW_BLOCK_ARMS == () and N.SHORT_ARMS == () and N.ADMISSION_PROBE_ARMS == () and N.K_OF["K4"] == 4 and N.K_OF["V65"] == 3)
M1.install(N, ARMS)
ok("M2 install is idempotent", N.ORDER == M1.ARM_KEYS)

# ---------------------------------------------------------------- M3 launch_check
Q["draft_head"] = "int8"
Q["expect_cfg"]["draft_head"] = "int8"
for k, a in ARMS.items():
    if k != "Q":
        M1.apply_head(N, a, True)
    bad = N.launch_check(a, env_of(a), with_head(a, a["draft_head"]), PIN)
    ok(f"M3 launch_check passes {k}'s planned launch (int8 head; I4 int4)", bad == [], bad)
ok("M3 apply_head: I4 int4, the others Q's int8; K4 / DYN trailing spec carries it",
   ARMS["I4"]["draft_head"] == "int4" and ARMS["V65"]["draft_head"] == "int8"
   and T.spec_from_argv(ARMS["DYN"]["extra"]).get("draft_lm_head_quant") == "int8")
bf = copy.deepcopy(ARMS)
for k in ("K4", "V32", "I4"):
    M1.apply_head(N, bf[k], False)
ok("M3 Q_I8 dropped -> levers BF16 (K4's trailing spec without the head), I4 stays int4; launch_check passes; check_arms exact",
   bf["K4"]["draft_head"] is None and "draft_lm_head_quant" not in T.spec_from_argv(bf["K4"]["extra"]) and bf["I4"]["draft_head"] == "int4"
   and N.launch_check(bf["K4"], env_of(bf["K4"]), with_head(bf["K4"], None), PIN) == []
   and N.launch_check(bf["V32"], env_of(bf["V32"]), with_head(bf["V32"], None), PIN) == [])
for k, var, val in (("Q", "VLLM_GDN_DECODE_FASTPATH", "1"), ("G2", "VLLM_GDN_DECODE_FASTPATH", "1"), ("G1", "VLLM_DRAFT_VOCAB_FILE", "/x.pt"),
                    ("Q", "VLLM_DRAFT_VOCAB_FILE", "/x.pt"), ("V32", "VLLM_DRAFT_VOCAB_SHA256", "0" * 64)):
    a = ARMS[k]
    bad = N.launch_check(a, env_of(a, **{var: val}), with_head(a, a["draft_head"]), PIN)
    ok(f"M3 launch_check refuses {var} on {k}", any(var in x for x in bad), bad)
bad = N.launch_check(ARMS["I4"], env_of(ARMS["I4"]), with_head(ARMS["I4"], "int8"), PIN)
ok("M3 launch_check refuses I4 with an int8 head", any("spec" in x for x in bad), bad)
bad = N.launch_check(ARMS["K4"], env_of(ARMS["K4"]), with_head(Q, "int8"), PIN)
ok("M3 launch_check refuses K4 launched with Q's argv (K 3, Q captures)", any("spec" in x for x in bad) and any("captures" in x for x in bad), bad)
bad = N.launch_check(ARMS["S28"], env_of(ARMS["S28"]), with_head(Q, "int8"), PIN)
ok("M3 launch_check refuses S28 launched at seqs 24", any("max-num-seqs" in x for x in bad), bad)


# ---------------------------------------------------------------- M4 read-backs
def env_raw(**vals):
    return {"results": [{"rank": r, "env": dict(vals)} for r in range(4)]}


def head_lines(mode, n):
    return "\n".join(f"(Worker_PP1_TP{r}_EP{r} pid=1) INFO 11-02 [draft_lm_head_quant.py:118] Draft lm_head: {mode} weight-only copy of the "
                     f"{n}x2560 shard (307.9 MiB; the target keeps its torch.bfloat16 head)" for r in (0, 1))


def vocab_log(n, path):
    return "\n".join(f"(Worker_PP1_TP{r}_EP{r} pid=1) INFO 11-02 [mtp.py:741] REDUCED DRAFT VOCAB ACTIVE: tp_rank={r} N={n} of 248320 "
                     f"(file {path}, table n={n}:sha256=ab), lm_head local shard ({n // 2}, 2560) torch.bfloat16 on cuda:0, has_own_lm_head=True"
                     for r in (0, 1))


def nondef(seqs=24, k=3, head="int8", extra=""):
    sp = "{'method': 'mtp', 'num_speculative_tokens': %d, 'draft_sample_method': 'probabilistic'%s%s}" % (
        k, ", 'draft_lm_head_quant': '%s'" % head if head else "", extra)
    return (f"(APIServer pid=1) INFO 11-02 [api_utils.py:286] non-default args: {{'model': '/tmp/fn_klee100', 'max_num_batched_tokens': 1024, "
            f"'max_num_seqs': {seqs}, 'speculative_config': {sp}, 'language_model_only': True}}")


def crb_of(a, log):
    return N.config_readback(a, log, PIN)


def rb(k, log, env=None, workers=None, head=None):
    a = ARMS[k]
    hd = a["draft_head"] if head is None else head
    i8 = T_I8(log, hd)
    return M1.readback(a, log, env if env is not None else env_raw(**{x: y for x, y in a["env"].items() if x in M1.M1_ENV_KEYS}),
                       crb_of(a, log), workers or {}, i8)


def T_I8(log, want):
    """The E28 I8 read-back's label logic on a log (the notebook passes e28k100_lib.i8_readback; here: its rule)."""
    heads = [{"mode": m.group(1), "n": int(m.group(2))} for m in re.finditer(r"Draft lm_head: (INT8|INT4) weight-only copy of the (\d+)x", log)]
    modes = sorted({h["mode"] for h in heads})
    if want:
        lab = "I8_ENGAGED" if modes == [want.upper()] else ("I8_CONTRADICTION" if heads else "I8_SPEC_ONLY")
    else:
        lab = "BF16_HEAD_CONTRADICTION" if heads else "BF16_HEAD"
    return {"label": lab, "draft_head_lines": heads}


base_log = nondef() + "\n" + head_lines("INT8", 124160)
for k in ("G1", "G2"):
    a = ARMS[k]
    log = base_log
    ok(f"M4 {k}: its GDN switch in 4 workers -> ok", rb(k, log)["ok"] is True, rb(k, log)["why"])
    ok(f"M4 {k}: the switch missing in a worker -> real failure", rb(k, log, env=env_raw())["real_failure"] is True)
    ok(f"M4 {k}: 3 workers -> real failure",
       rb(k, log, env={"results": env_raw(**a["m1_change"]["env"])["results"][:3]})["real_failure"] is True)
ok("M4 IDX with a GDN switch in the workers -> real failure (absent elsewhere)",
   rb("G1", base_log, env=env_raw(VLLM_GDN_DECODE_FASTPATH="1", VLLM_GDN_SHARED_ALIGN_INDEX="1"))["real_failure"] is True)
for k, (name, sh, n) in M1.VOCAB_TABLES.items():
    good = nondef() + "\n" + vocab_log(n, VPATH[k]) + "\n" + head_lines("INT8", n // 2)
    ok(f"M4 {k}: REDUCED DRAFT VOCAB on both PP1 ranks (N {n}, the planned file) + the INT8 head on the reduced shard -> ok",
       rb(k, good)["ok"] is True, rb(k, good)["why"])
    ok(f"M4 {k}: no REDUCED DRAFT VOCAB line -> real failure", rb(k, base_log)["real_failure"] is True)
    ok(f"M4 {k}: the INT8 head on the FULL shard (vocab not under the quantized head) -> real failure",
       rb(k, nondef() + "\n" + vocab_log(n, VPATH[k]) + "\n" + head_lines("INT8", 124160))["real_failure"] is True)
    ok(f"M4 {k}: another table's N -> real failure", rb(k, nondef() + "\n" + vocab_log(16384, VPATH[k]) + "\n" + head_lines("INT8", 8192))["real_failure"] is True)
ok("M4 a reduced vocab on G1 -> real failure", rb("G1", base_log + "\n" + vocab_log(65536, "/x.pt"))["real_failure"] is True)
ok("M4 I4: INT4 head lines on both PP1 ranks -> ok; INT8 -> real failure",
   rb("I4", nondef(head="int4") + "\n" + head_lines("INT4", 124160))["ok"] is True
   and rb("I4", base_log)["real_failure"] is True)
ok("M4 a lever with no head build line -> real failure (head not read back)", rb("G2", nondef())["real_failure"] is True)
W_FP8 = {"ranks": [{"rank": r, "kv_cache_specs": [{"type": "MLAAttentionSpec", "example": "language_model.model.layers.3.self_attn.indexer.compressed_key_cache",
                                                   "dtype": "torch.float8_e4m3fn"}]} for r in range(4)]}
W_BF16 = {"ranks": [{"rank": r, "kv_cache_specs": [{"type": "MLAAttentionSpec", "example": "x.compressed_key_cache", "dtype": "torch.bfloat16"}]} for r in range(4)]}
ok("M4 IDX: compressed_key_cache fp8 on every rank -> ok; bf16 -> real failure; no spec + logged fp8 -> ok; neither -> real failure",
   rb("IDX", base_log, workers=W_FP8)["ok"] is True and rb("IDX", base_log, workers=W_BF16)["real_failure"] is True
   and rb("IDX", base_log.replace("'language_model_only': True", "'attention_config': {'indexer_kv_dtype': 'fp8'}, 'language_model_only': True"))["ok"] is True
   and rb("IDX", base_log)["real_failure"] is True)
ok("M4 the fp8 indexer on G1 -> real failure", rb("G1", base_log, workers=W_FP8)["real_failure"] is True)
for k, s in M1.SEQ_ARMS.items():
    ok(f"M4 {k}: logged max_num_seqs {s} -> ok; 24 -> real failure", rb(k, nondef(seqs=s) + "\n" + head_lines("INT8", 124160))["ok"] is True
       and rb(k, base_log)["real_failure"] is True)
lp_log = (nondef().replace("'max_num_batched_tokens': 1024", "'max_num_batched_tokens': 2048, 'long_prefill_token_threshold': 1024, "
                           "'long_prefill_token_threshold_adaptive': True") + "\n" + head_lines("INT8", 124160))
ok("M4 LP: logged 2048 / 1024 / adaptive -> ok; Q's 1024 -> real failure; LP values on G2 -> real failure",
   rb("LP", lp_log)["ok"] is True and rb("LP", base_log)["real_failure"] is True and rb("G2", lp_log)["real_failure"] is True)
k4_log = nondef(k=4) + "\n" + head_lines("INT8", 124160)
dyn_log = nondef(k=4, extra=", 'num_speculative_tokens_per_batch_size': [[1, 8, 4], [9, 24, 3]]") + "\n" + head_lines("INT8", 124160)
ok("M4 K4: logged K 4 -> ok; K 3 -> real failure", rb("K4", k4_log)["ok"] is True and rb("K4", base_log)["real_failure"] is True)
ok("M4 DYN: logged K 4 + the schedule -> ok; no schedule -> real failure; the PIECEWISE override -> real failure",
   rb("DYN", dyn_log)["ok"] is True and rb("DYN", k4_log)["real_failure"] is True
   and rb("DYN", dyn_log + "\n(EngineCore pid=1) WARNING [vllm.py:1079] Dynamic speculative decoding changes the target verification length "
          "at runtime. Overriding cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE for reliability.")["real_failure"] is True
   and rb("DYN", dyn_log + "\nWARNING Disabling num_speculative_tokens_per_batch_size and falling back")["real_failure"] is True)
ok("M4 a dynamic schedule on K4 -> real failure", rb("K4", dyn_log)["real_failure"] is True)
ok("M4 K 4 on G1 -> real failure", rb("G1", k4_log)["real_failure"] is True)
qprof = {"status": "DONE", "steady": {"profile_prof_8k@16": {"trace_files": [["a.pt.trace.json.gz", 9]]},
                                      "profile_prof_20k@24": {"trace_files": [["b.pt.trace.json.gz", 9]]}}}
ok("M4 PROF: both 30-s windows with trace files -> DONE; one empty -> NOT_INTEGRATED; switch off / Q off / not run",
   M1.prof_record(qprof, M1.switch_map(None))["status"] == "DONE"
   and M1.prof_record({"steady": {"profile_prof_8k@16": {"trace_files": []}, "profile_prof_20k@24": {"trace_files": [["b", 1]]}}},
                      M1.switch_map(None))["status"] == "NOT_INTEGRATED"
   and M1.prof_record(qprof, M1.COPIES["a"])["status"] == "SKIPPED_SWITCH_OFF"
   and M1.prof_record(qprof, dict(M1.COPIES["c"], Q=False))["status"] == "SKIPPED_DEPENDENCY"
   and M1.prof_record({"status": "LAUNCH_FAILED"}, M1.switch_map(None))["status"] == "NOT_RUN")

# ---------------------------------------------------------------- M5 switches, copies, the time plan
ok("M5 switch_map: None -> all on; a dict -> a missing key off; not a dict -> all on (recorded)",
   all(M1.switch_map(None).values()) and M1.switch_map({"Q": True})["V65"] is False and all(M1.switch_map("x").values())
   and M1.switch_record("x")["invalid_not_a_dict"] is True and M1.switch_record({"Q": True, "ZZ": 1})["unknown_keys"] == ["ZZ"])
ok("M5 the three copies: Q in each; every lever + PROF in exactly one copy; A = V65 V32 I4, B = IDX S28 S32 LP, C = G1 G2 K4 DYN PROF",
   all(c["Q"] for c in M1.COPIES.values()) and all(sum(c[k] for c in M1.COPIES.values()) == 1 for k in M1.LEVERS + ("PROF",))
   and {k for k, v in M1.COPIES["a"].items() if v} == {"Q", "V65", "V32", "I4"} and {k for k, v in M1.COPIES["b"].items() if v} == {"Q", "IDX", "S28", "S32", "LP"}
   and {k for k, v in M1.COPIES["c"].items() if v} == {"Q", "G1", "G2", "K4", "DYN", "PROF"} and all(set(c) == set(M1.SWITCH_KEYS) for c in M1.COPIES.values()))
ok("M5 owners: a foysalemonshanto, b esfersami50, c foysalemonshanto", M1.OWNERS == {"a": "foysalemonshanto", "b": "esfersami50", "c": "foysalemonshanto"})
W = {x: M1.copy_estimate(ARMS, sw) for x, sw in M1.COPIES.items()}
ok("M5 every copy's planned walk runs all its arms within the hard stop", all(w["all_run"] for w in W.values()), {x: w["walk"] for x, w in W.items()})
ok(f"M5 every copy's runtime <= {M1.COPY_RUNTIME_MAX} min; BUDGET_MIN = max walk + 5 rounded up to 5; deadman + 5",
   all(w["runtime_min"] <= M1.COPY_RUNTIME_MAX for w in W.values())
   and M1.BUDGET_MIN == 5 * -(-max(w["runtime_min"] + M1.START_RESERVE_MIN for w in W.values()) // 5) and M1.DEADMAN_MIN == M1.BUDGET_MIN + 5,
   {x: w["runtime_min"] for x, w in W.items()})
ok("M5 copy C's Q carries PROF's minutes, copy A's / B's does not", W["c"]["walk"][0]["est_min"] == M1.q_time(True)[0]
   and W["a"]["walk"][0]["est_min"] == M1.q_time(False)[0] and M1.q_time(True)[0] > M1.q_time(False)[0])
ok("M5 every lever's est_min = ready + body + teardown of its own plan; cap = body + 1 min",
   all(a["est_min"] == M1.lever_time(k)[0] and a["cap_min"] == M1.lever_time(k)[1] for k, a in ARMS.items() if k != "Q"))

# ---------------------------------------------------------------- M6 the built body
BODY = HERE / "nbm1_cell7_body.py"
if BODY.is_file():
    btxt = BODY.read_text(encoding="utf-8")
    if (HERE / "nbm1_export_cells.json").is_file():   # the units kit: the export's cells (CELL 7 stub omitted: size), sha-checked
        _cells = json.loads((HERE / "nbm1_export_cells.json").read_text(encoding="utf-8"))
        _fx = json.loads((HERE / "nbm1_build_facts.json").read_text(encoding="utf-8"))
        import hashlib as _h2
        ok("M6 the kit's export cells = the build's cells (sha per cell; only the CELL 7 stub omitted)",
           len(_cells) == len(_fx["nbdf_code_cell_sha256"]) and sum(c is None for c in _cells) == 1
           and all(c is None or _h2.sha256(c.encode()).hexdigest() == s for c, s in zip(_cells, _fx["nbdf_code_cell_sha256"]))
           and _h2.sha256(btxt.encode()).hexdigest() == _fx["body_file_sha256"])
        etxt = "\n\n".join(c if c is not None else "# CELL 7 stub omitted in the units kit (it execs nbm1_cell7_body.py)" for c in _cells)
    else:
        etxt = (HERE / "nbm1_code_export.py").read_text(encoding="utf-8")
    fns = {n.name: ast.get_source_segment(btxt, n) for n in ast.parse(btxt).body if isinstance(n, ast.FunctionDef)}
    for s_ in ("M1.m1_arms(N, _nbm1_q0, M1_SWITCH, M1_VOCAB)", "M1.install(N, ARMS)", "M1.check_arms(N, T, ARMS, _nbm1_q0)",
               "M1.readback(arm, t,", "M1.pre_gate(a, KV_KIT_GATE)", "M1.apply_head(N, arm, Q_I8[\"on\"])", '",".join(M1.M1_ENV_KEYS)',
               "if not is_d:                              # E25 NB-M1", "M1.arm_records(", "M1.prof_record(q, M1_SWITCH)",
               'arm.get("windows_fixed") or K100_CFG["windows"]', 'secs=float(M1.PROF_SECS)', "if (is_d or is_lv) and o[\"smoke_gate\"].startswith(\"FAIL\")",
               'arm.get("m1_no_ladder")', 'M1.item_on(arm, "tf")', 'M1.item_on(arm, "long_oracle")', 'M1.item_on(arm, "prefix")',
               '(band in ("40k",) + N.LONG_BANDS or arm.get("m1_lever"))', 'def profile_window(port, arm, band="8k", n=None, secs=6.0):'):
        ok(f"M6 body has {s_}", s_ in btxt)
    ok("M6 the D loop is gone from nbdf_main (the NB-M1 loop instead)", "for k in N.D_ARMS:" not in fns["nbdf_main"] and "for k in M1.LEVERS:" in fns["nbdf_main"])
    ok("M6 OOM at any stage (incl. the first request after ready) steps the ladder (GPU v1 fix)",
       'oom = bool(o["oom_in_log"]) and not o["xid_in_log"]' in btxt and 'oom = o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]' not in btxt)
    ok("M6 body reads the switch (M1_SWITCH from NBM1_ON); the switch line is NOT in the body", 'globals().get("NBM1_ON")' in btxt
       and "NBM1_ON =" not in btxt and "SKIPPED_SWITCH_OFF" in btxt)
    sw_ = [l for l in etxt.splitlines() if l.startswith("NBM1_ON = ")]   # the export = the notebook's code cells
    ok("M6 the code export has ONE plain NBM1_ON line (CELL 1, default: every arm on)", len(sw_) == 1
       and ast.literal_eval(sw_[0][len("NBM1_ON = "):]) == M1.SWITCH_DEFAULT and etxt.index(sw_[0]) < etxt.index("# ==================================================================== CELL 1"), sw_)
    i_ = etxt.index('K100_CFG = json.loads(r"""') + len('K100_CFG = json.loads(r"""')
    k100cfg = json.loads(etxt[i_:etxt.index('"""', i_)])
    ok("M6 K100_CFG in the notebook (CELL 2b): q_i8 True (Q = E28 production), seqs 24, pin 7,933,526,016, the Q ladder, the K100 windows",
       k100cfg["q_i8"] is True and k100cfg["seqs"] == 24 and k100cfg["q_pin"] == PIN and k100cfg["q_ladder"] == [7933526016, 7665090560]
       and k100cfg["windows"] == M1.Q_WINDOWS, k100cfg.get("q_i8"))
    m_ = re.search(r'BUDGET = int\(os.environ.get\("RC2_BUDGET_SEC"\) or (\d+) \* 60\)', etxt)
    md_ = re.search(r"NBDF_DEADMAN_SEC = (\d+)", etxt)
    ok(f"M6 CELL 1 hard stop {M1.BUDGET_MIN} min, deadman {M1.DEADMAN_MIN} min", m_ and int(m_.group(1)) == M1.BUDGET_MIN and md_
       and int(md_.group(1)) == M1.DEADMAN_MIN * 60, (m_ and m_.group(1), md_ and md_.group(1)))
    FACTS_P = HERE / "nbm1_build_facts.json"
    if FACTS_P.is_file():
        fx = json.loads(FACTS_P.read_text(encoding="utf-8"))
        cps = fx.get("copies") or {}
        ok("M6 build facts: three copies, each switch line = repr(COPIES[x]), owners, the vocab dataset only on copy A",
           set(cps) == {"a", "b", "c"} and all(cps[x]["switch_line"] == "NBM1_ON = " + repr(M1.COPIES[x]) and cps[x]["owner"] == M1.OWNERS[x]
                                                for x in cps)
           and (M1.VOCAB_DS in cps["a"]["dataset_sources"]) and M1.VOCAB_DS not in cps["b"]["dataset_sources"] + cps["c"]["dataset_sources"], cps.keys())

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

    def main_ns(q_rec=None, d_out=None, gate=True, switch=None, arms=None, seam_ok=True):
        calls, R_ = [], {"sections": {"A0": {"merge": {"ok": True}}}, "setup": {}}
        arms_ = arms or ARMS

        def runner(k):
            calls.append(k)
            if k == "Q":
                return dict(q_rec or {"status": "DONE", "pin": PIN})
            return dict((d_out or {}).get(k) or {"status": "DONE", "pin": PIN, "smoke_gate": "PASS"})

        def step(sec, name, est, fn):
            r_ = dict(fn() or {})
            if r_.get("status") in (None, "NOT_RUN"):
                r_["status"] = "DONE"
            R_["sections"].setdefault(sec, {})[name] = r_
            return r_
        ns = {"N": N, "P": P, "T": T, "M1": M1, "R": R_, "ARMS": arms_, "ISO": {}, "NBDF": {}, "FAKE": False, "ROWS": {"rc2_fp8": 1},
              "K100_CFG": k100cfg, "PENDING": [], "step": step, "save": lambda: None, "log": lambda *a, **k: None, "finalize": lambda: None,
              "left": lambda: 10 * 3600, "merlin_install": lambda: {"kv": {"MERLIN_TREE": "/x"}},
              "apply_60490": lambda: {"status": "DONE", "ok": True}, "k100_setup": lambda: {"ok": True}, "sha_gate": lambda t: (True, "ok"),
              "SEAM_API": object(), "SEAM_LOAD": {}, "SEAM_ENVS": {}, "_tree_py": lambda: (_ for _ in ()).throw(RuntimeError("no tree")),
              "PIN": {"kv_bytes": PIN}, "D_PIN": {"kv_bytes": None}, "SW_PIN": {"kv_bytes": None}, "Q_REC": {},
              "DRAFTER": {"ok": True, "status": None, "why": "x"},
              "KV_KIT_GATE": {"passed": gate, "per_k": {}, "why": None if gate else ["closed"]},
              "_AKEYS": ("name", "est_min"), "PSEQ": None, "SSEQ": None, "run_arm": runner,
              "M1_SWITCH": M1.switch_map(switch)}
        for fn_ in ("_arm", "skip_arm", "_patch_rec", "_seam_rec", "nbdf_main", "_k100_run_pending", "_k100_tap_arm"):
            exec(fns[fn_], ns)
        ns["nbdf_main"](runner=runner, pseq=_Seq(), sseq=_Seq(seam_ok))
        return calls, R_, ns

    def status_of(R_, k):
        return next(iter(R_["sections"].get(k, {"": {}}).values())).get("status")

    c_, R_, ns_ = main_ns()
    ok("M6 nbdf_main (every arm on): Q then V65 V32 I4 IDX S28 S32 LP G1 G2 K4 DYN", c_ == list(M1.ARM_KEYS), c_)
    for x, sw in M1.COPIES.items():
        c_, R_, ns_ = main_ns(switch=sw)
        want = [k for k in M1.ARM_KEYS if sw[k]]
        ok(f"M6 copy {x.upper()}: exactly its arms launch; every other lever SKIPPED_SWITCH_OFF", c_ == want
           and all(status_of(R_, k) == "SKIPPED_SWITCH_OFF" for k in M1.LEVERS if not sw[k]), (c_, want))
        ok(f"M6 copy {x.upper()}: PROF recorded ({'run inside Q' if sw['PROF'] else 'SKIPPED_SWITCH_OFF'})",
           R_["sections"]["PROF"]["nbm1_prof"]["status"] == ("NOT_RUN" if sw["PROF"] else "SKIPPED_SWITCH_OFF"), R_["sections"].get("PROF"))
    c_, R_, ns_ = main_ns(switch=M1.COPIES["c"], q_rec=qprof)
    ok("M6 copy C with Q's PROF windows traced -> PROF DONE", R_["sections"]["PROF"]["nbm1_prof"]["status"] == "DONE")
    c_, R_, ns_ = main_ns(switch=dict(M1.COPIES["b"], Q=False))
    ok("M6 Q switched off: Q SKIPPED_SWITCH_OFF, copy B's levers still launch", c_ == ["IDX", "S28", "S32", "LP"] and status_of(R_, "Q") == "SKIPPED_SWITCH_OFF", c_)
    for nm_, q_ in (("LAUNCH_FAILED", {"status": "LAUNCH_FAILED"}), ("NO_GO_QUALITY", {"status": "NO_GO_QUALITY"})):
        c_, R_, ns_ = main_ns(q_rec=q_, switch=M1.COPIES["a"])
        ok(f"M6 Q {nm_} -> every lever of the copy still launches (no cross-arm skip)", c_ == ["Q", "V65", "V32", "I4"], c_)
    c_, R_, ns_ = main_ns(switch=M1.COPIES["c"], d_out={"G1": {"status": "SERVER_DIED"}, "K4": {"status": "NOT_INTEGRATED", "not_integrated": True}})
    ok("M6 a failed lever never skips the next (then the K100 deferred Q-defer-off rerun of the died arm: G1_ISO)",
       c_[:5] == ["Q", "G1", "G2", "K4", "DYN"] and c_[5:] == ["G1_ISO"], c_)
    c_, R_, ns_ = main_ns(gate=False)
    ok("M6 kit gate closed (the closed build) -> only Q launches; every lever SKIPPED_KV_KIT_GATE",
       c_ == ["Q"] and all(status_of(R_, k) == "SKIPPED_KV_KIT_GATE" for k in M1.LEVERS), c_)
    c_, R_, ns_ = main_ns(arms=NOV)
    ok("M6 vocab tables missing -> V65 / V32 NOT_INTEGRATED (never booted), the others launch",
       c_ == [k for k in M1.ARM_KEYS if k not in ("V65", "V32")] and status_of(R_, "V65") == "NOT_INTEGRATED", c_)
    c_, R_, ns_ = main_ns(seam_ok=False, switch=M1.COPIES["c"])
    ok("M6 the dflash seam failing to apply never stops an MTP arm", c_ == ["Q", "G1", "G2", "K4", "DYN"], c_)

    # arm_boot with fakes: the OOM ladder (any stage), the lever head into build()
    class _Srv:
        def __init__(self, log):
            self._log, self.ready_sec, self.argv = log, 1.0, []

        def log(self):
            return self._log

    def boot_ns(outcomes, arm, q_i8=True, left_sec=10 * 3600):
        tries, seen = list(outcomes), []
        ns = {"N": N, "T": T, "M1": M1, "re": re, "K100_CFG": k100cfg, "Q_I8": {"on": q_i8, "want": True, "dropped": None},
              "NB4T_BUILD_ENV": {}, "NB1_EXTRA": {}, "NB1_ENV": {}, "NB1_KV": {}, "ARM_OF": {}, "CUR": {}, "BOOT": {}, "RESP": {},
              "NBDF_READY_SEC": 1200, "left": lambda: left_sec, "log": lambda *a, **k: None, "tree_check": lambda a: {"ok": True},
              "boot_gate": lambda: {"ok": True}, "shm_names": lambda: [], "shm_cleanup": lambda x: {}, "q_readback": lambda s, a: {},
              "OOM_ERR": re.compile(r"CUDA out of memory|OutOfMemoryError"), "XID": re.compile(r"Xid \d+"),
              "gated": lambda o, need=(), expected_rows=4: o, "dflash_rb": lambda a, t: {}, "arm_watched": lambda b, c, n: b}

        class _Mon:
            def __init__(self, n):
                self.last, self.stop_ev = None, type("E", (), {"set": lambda s: None})()

            def start(self):
                pass
        ns["LoadMonitor"] = _Mon

        def mfs(name, kv, port, kvb):
            seen.append({"name": name, "port": port, "kvb": kvb, "build_env": dict(ns["NB4T_BUILD_ENV"].get(name) or {}),
                         "extra": list(ns["NB1_EXTRA"].get(name) or [])})
            return _Srv(tries[min(len(seen) - 1, len(tries) - 1)][1])
        ns["merlin_fn_server"] = mfs
        ns["with_servers"] = lambda srvs, ready, b: {"status": tries[min(len(seen) - 1, len(tries) - 1)][0]}
        for fn_ in ("_k100_retry", "arm_boot"):
            exec(fns[fn_], ns)
        o = ns["arm_boot"](arm, lambda ss: {})
        return o, seen, ns

    a = copy.deepcopy(ARMS["G1"])
    o, seen, ns = boot_ns([("SERVER_DIED", "torch.OutOfMemoryError: CUDA out of memory (first request after ready)"), ("DONE", "ok")], a)
    ok("M6 arm_boot: an OOM after ready (status SERVER_DIED) -> a second try one rung lower (7,665,090,560), port + 1",
       [s_["kvb"] for s_ in seen] == [PIN, 7665090560] and seen[1]["port"] == seen[0]["port"] + 1 and len(o["tries"]) == 2, seen)
    o, seen, ns = boot_ns([("SERVER_DIED", "CUDA out of memory\nNVRM: Xid 79"), ("DONE", "ok")], copy.deepcopy(ARMS["G1"]))
    ok("M6 arm_boot: OOM with an Xid (device fault) -> no retry", len(seen) == 1, seen)
    o, seen, ns = boot_ns([("LAUNCH_FAILED", "CUDA out of memory")] * 3, copy.deepcopy(ARMS["G2"]))
    ok("M6 arm_boot: the Q ladder ends after its last rung (2 tries)", [s_["kvb"] for s_ in seen] == [PIN, 7665090560], seen)
    o, seen, ns = boot_ns([("DONE", "ok")], copy.deepcopy(ARMS["I4"]))
    ok("M6 arm_boot: I4 -> DRAFT_HEAD=int4 into Merlin's build()", seen[0]["build_env"] == {"DRAFT_HEAD": "int4"}, seen)
    o, seen, ns = boot_ns([("DONE", "ok")], copy.deepcopy(ARMS["V65"]))
    ok("M6 arm_boot: V65 -> Q's DRAFT_HEAD=int8 into build(), the vocab env in NB1_ENV", seen[0]["build_env"] == {"DRAFT_HEAD": "int8"}
       and ns["NB1_ENV"]["nbm1_v65"][M1.ENV_VOCAB] == VPATH["V65"], seen)
    a = copy.deepcopy(ARMS["K4"])
    o, seen, ns = boot_ns([("DONE", "ok")], a, q_i8=False)
    ok("M6 arm_boot: Q_I8 dropped -> K4 BF16 head (no DRAFT_HEAD, trailing spec without the head)", seen[0]["build_env"] == {}
       and "draft_lm_head_quant" not in T.spec_from_argv(seen[0]["extra"]) and T.spec_from_argv(seen[0]["extra"])["num_speculative_tokens"] == 4, seen)
    a = copy.deepcopy(Q)
    o, seen, ns = boot_ns([("DONE", "ok")], a)
    ok("M6 arm_boot: Q keeps the K100 I8 path (DRAFT_HEAD=int8)", seen[0]["build_env"] == {"DRAFT_HEAD": "int8"} and a["draft_head"] == "int8", seen)

    # run_window (fakes): a KV-bound lever window steps one level down on any band (S28 / S32 at 20k: never skipped)
    def win_ns(arm_, first_kv_bound=True):
        calls = []

        def hw(port, s, band, n, arm):
            calls.append(n)
            kb = first_kv_bound and len(calls) == 1
            return {"status": "CAPACITY_LIMITED" if kb else "STEADY", "level": n, "running_max": n - 1 if kb else n, "tok_s": 1.0}

        class _S:
            port = 1

            def alive(self):
                return True
        ns = {"N": N, "CUR": {}, "due": lambda sec: True, "boot_left": lambda: 1e9, "win_need": lambda b, n: 10, "hist_window": hw,
              "_live": lambda o, k, v: o.__setitem__(k, v), "traceback": __import__("traceback")}

        class _W:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        ns["_watched"] = _W
        exec(fns["run_window"], ns)
        o = {"steady": {}, "items_skipped": [], "long_levels": {}}
        ns["run_window"](o, _S(), arm_, "20k", 28, "")
        return calls, o
    calls_, o_ = win_ns(ARMS["S28"])
    ok("M6 run_window: S28's KV-bound 20k@28 -> one step down to 27 (recorded), never skipped",
       calls_ == [28, 27] and "20k@27" in o_["steady"] and o_["residency_retry"]["20k"] == [28, 27], (calls_, list(o_["steady"])))
    calls_, o_ = win_ns(Q)
    ok("M6 run_window: Q's 20k@24 keeps NBDF-K100's rule (no step down at 20k)", calls_ == [28], calls_)

    # the M7 verdict over the switched / skipped arms
    recs_sw = {"Q": {"status": "DONE"}, "V65": {"status": "DONE"}, "G1": {"status": "SKIPPED_SWITCH_OFF", "why": "x"},
               "IDX": {"status": "NOT_INTEGRATED", "not_integrated": True}}
    try:
        v_ = N.verdict(recs_sw, ARMS)
        vs_ = {k: a_["verdict"] for k, a_ in v_["arms"].items()}
    except Exception as e:
        vs_ = repr(e)
    ok("M7 verdict over the NB-M1 arms (Q = base): every lever judged, skipped / not integrated recorded, no exception",
       isinstance(vs_, dict) and set(vs_) == set(M1.LEVERS) and vs_["IDX"] == "NOT_INTEGRATED", vs_)
else:
    print("SKIP M6 the built body is not present")

Wq = {"8k@16#1": {"status": "STEADY", "tok_s": 500.0, "running_eq_level": True, "prefill_done_before_window": True, "level": 16},
      "8k@16#2": {"status": "STEADY", "tok_s": 510.0, "running_eq_level": True, "prefill_done_before_window": True, "level": 16},
      "90k@8": {"status": "STEADY", "tok_s": 340.0, "running_eq_level": True, "prefill_done_before_window": True, "level": 8}}
Wx = {"8k@16#1": {"status": "STEADY", "tok_s": 560.0, "running_eq_level": True, "prefill_done_before_window": True, "level": 16},
      "90k@8": {"status": "STEADY", "tok_s": 300.0, "running_eq_level": True, "prefill_done_before_window": True, "level": 8}}
try:
    bt = M1.band_table(T, {"steady": Wq}, {"steady": Wx}, {"90k": 0.037})
    btv = (bt["bands"]["8k"].get("label"), bt["bands"]["90k"].get("label"), bt["bands"]["116k"].get("label"))
except Exception as e:
    btv = repr(e)
ok("M7 band_table: 8k GAIN, 90k REGRESSION, 116k INCONCLUSIVE (records only)", btv == ("GAIN", "REGRESSION", "INCONCLUSIVE"), btv)
ok("M7 arm_records: per-arm raw numbers", M1.arm_records(T, {"status": "DONE", "steady": Wx}, "G1", {"steady": Wq})["status"] == "DONE")
print(f"{RES['pass']}/{RES['pass'] + RES['fail']} PASS", flush=True)
sys.exit(1 if RES["fail"] else 0)
