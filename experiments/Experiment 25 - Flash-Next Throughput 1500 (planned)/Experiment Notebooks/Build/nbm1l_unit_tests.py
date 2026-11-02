"""E25 NB-M1L (light) quick unit tests (local + the Py3.12 CPU kit), on the real nbdf_lib / nb4t_lib / nb4_ppack, nbm1_lib +
nbm1l_overlay (exec'd into nbm1_lib's namespace, as the notebook does) and the BUILT light body / export / facts:
  L1  the light plan: every lever = Q + exactly the NB-M1 change (env / argv / spec / captures / head / windows = the default
      NB-M1 plan); Q items: facts, smoke without the ladder, the prefill probe, warmup, windows; no oracle / oracle A/A / long
      oracle / prefix / TF / profiles; levers: no TF (IDX / LP included), no oracle
  L2  the time plan: due-aware caps (= the body's own win_need), copies B / C walks, runtime <= COPY_RUNTIME_MAX, the hard
      stop holds every arm's full cap + 5 min, BUDGET / DEADMAN in CELL 1
  L3  the built body: the overlay embedded = nbm1l_overlay.py (sha = facts), the light edits present, the switch line from the
      code EXPORT, the facts (copies B hosen42 / C foysalemonshanto, every kernel source kept)
  L4  arm_body (fakes): Q in copy B / C and every lever of B / C call ONLY their light items (removed items never called; PROF's
      two 30-s windows in C); a lever's smoke FAIL / read-back real failure still stops that arm
  L5  arm_boot (fakes): the light evidence gate (no oracle in the plan), the OOM ladder unchanged; nbdf_main (fakes): the two
      copies launch exactly their arms; Q off; the closed gate; a failed lever never skips the next
  L6  hard-only gates without an oracle + nbdf_lib.verdict over light records (labels INCONCLUSIVE, no exception)
Run: python nbm1l_unit_tests.py   -> 'FAIL <name>' lines, then '<pass>/<total> PASS'."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
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

BUILT = Path(os.environ.get("NBM1L_BUILT_DIR") or HERE)   # local: a --test-out folder; the kit: HERE
OVL_SRC = (HERE / "nbm1l_overlay.py").read_text(encoding="utf-8")
exec(compile(OVL_SRC, "nbm1l_overlay.py", "exec"), M1.__dict__)
_sp = importlib.util.spec_from_file_location("nbm1_lib_default", str(HERE / "nbm1_lib.py"))
D1 = importlib.util.module_from_spec(_sp)      # the default NB-M1 lib (no overlay): the plan reference
_sp.loader.exec_module(D1)

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
REMOVED_Q = ("ladder", "oracle", "oracle_aa", "long_oracle", "prefix", "profiles", "tf")

# ---------------------------------------------------------------- L1 the light plan
ARMS = M1.m1_arms(N, K100["Q"], None, VOCAB)
DEF = D1.m1_arms(N, copy.deepcopy(K100["Q"]), None, VOCAB)
Q = ARMS["Q"]
ok("L1 overlay: LIGHT, version tagged, copies B / C only, owners B hosen42 / C foysalemonshanto",
   M1.LIGHT is True and "nbm1l-overlay" in M1.LIB_VERSION and set(M1.COPIES) == {"b", "c"}
   and M1.OWNERS == {"b": "hosen42", "c": "foysalemonshanto"})
ok("L1 light copies: B = Q IDX S28 S32 LP, C = Q G1 G2 K4 DYN PROF (= NB-M1's B / C switch lines)",
   M1.COPIES == {"b": D1.COPIES["b"], "c": D1.COPIES["c"]}
   and {k for k, v in M1.COPIES["b"].items() if v} == {"Q", "IDX", "S28", "S32", "LP"}
   and {k for k, v in M1.COPIES["c"].items() if v} == {"Q", "G1", "G2", "K4", "DYN", "PROF"})
ok("L1 arms in order Q + the 11 levers", tuple(ARMS) == M1.ARM_KEYS == D1.ARM_KEYS)
try:
    M1.check_arms(N, T, ARMS, Q0)
    ca = True
except ValueError as e:
    ca = repr(e)
ok("L1 check_arms: light Q = K100 Q, every lever = Q + exactly its change", ca is True, ca)
PLAN_FIELDS = ("env", "extra", "spec", "captures", "k", "kv_bytes", "port", "q_label", "expect", "expect_cfg", "coverage", "streams",
               "m1_change", "m1_head", "windows_fixed", "name", "m1_vocab", "m1_trailing_spec", "ring_needed")
for k in ARMS:
    diff = [f for f in PLAN_FIELDS if ARMS[k].get(f) != DEF[k].get(f)]
    ok(f"L1 {k}: the change / argv / env / spec / captures / windows = the default NB-M1 plan (only items + time differ)", not diff, diff)
ok("L1 Q items: prefill probe on (the windows' admission rate); ladder, oracle, oracle A/A, long oracle, prefix, profiles, TF off",
   M1.item_on(Q, "prefill_probe") and not any(M1.item_on(Q, x) for x in REMOVED_Q) and Q["m1_no_ladder"] is True)
ok("L1 the default NB-M1 Q keeps every item (reference)", all(D1.item_on(DEF["Q"], x) for x in REMOVED_Q) and not DEF["Q"].get("m1_no_ladder"))
ok("L1 every lever: no TF (IDX / LP included), no oracle, none of Q's record-only items; ladder skipped",
   all(not M1.item_on(a, x) for k, a in ARMS.items() if k != "Q" for x in REMOVED_Q + ("prefill_probe",))
   and all(a["m1_no_ladder"] for k, a in ARMS.items() if k != "Q"))
ok("L1 the default NB-M1 had TF on IDX / LP and the oracle on every lever (reference)",
   {k for k, a in DEF.items() if k != "Q" and D1.item_on(a, "tf")} == {"IDX", "LP"} and all(D1.item_on(a, "oracle") for a in DEF.values()))
ok("L1 every lever's windows keep 90k@8 and 116k@6; Q keeps its six windows (8k@16 x 2, 20k@24, 40k@16, 90k@8, 116k@6)",
   all(any(w[:2] == ["90k", 8] for w in a["windows_fixed"]) and any(w[:2] == ["116k", 6] for w in a["windows_fixed"])
       for k, a in ARMS.items() if k != "Q") and "windows_fixed" not in Q
   and M1.Q_WINDOWS == [["8k", 16, "#1"], ["8k", 16, "#2"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]])
try:
    exec(compile(OVL_SRC, "nbm1l_overlay.py", "exec"), dict(M1.__dict__))
    _twice = "no raise"
except AssertionError as e:
    _twice = str(e)
ok("L1 the overlay refuses a second application", "applied twice" in _twice, _twice)

# ---------------------------------------------------------------- L2 the time plan
qb, qc = M1.q_time(False), M1.q_time(True)
ok("L2 Q (copy B, no PROF) 37 min, cap 29; Q (copy C, PROF) 48 min, cap 40", (qb[:2], qc[:2]) == ((37, 29), (48, 40)), (qb[:2], qc[:2]))
ok("L2 Q's light body = facts + smoke + prefill probe + warmup + the six windows + end (+ PROF's two windows in C)",
   [r[0] for r in qb[2]["items"]] == ["facts", "smoke", "prefill_probe", "warmup", "8k@16#1", "8k@16#2", "20k@24", "40k@16", "90k@8", "116k@6", "end"]
   and [r[0] for r in qc[2]["items"]][-3:] == ["prof_8k@16", "prof_20k@24", "end"], [r[0] for r in qb[2]["items"]])
WANT_L = {"V65": (31, 27), "V32": (31, 27), "I4": (31, 27), "IDX": (24, 20), "S28": (25, 22), "S32": (26, 23), "LP": (24, 20),
          "G1": (22, 20), "G2": (22, 20), "K4": (24, 20), "DYN": (24, 20)}
ok("L2 lever est / cap (min)", {k: M1.lever_time(k)[:2] for k in M1.LEVERS} == WANT_L, {k: M1.lever_time(k)[:2] for k in M1.LEVERS})
ok("L2 every arm's est / cap in the plan = the time model", Q["est_min"] == qc[0] and Q["cap_min"] == qc[1]
   and all((a["est_min"], a["cap_min"]) == M1.lever_time(k)[:2] for k, a in ARMS.items() if k != "Q"))
cap_ok = []
for k, a in list(ARMS.items()) + [("Q_B", dict(Q, cap_min=qb[1], time_model=qb[2]))]:
    tb = a["time_model"]
    worst = max(r[3] + r[2] for r in tb["items"] if r[2] is not None)
    cap_ok.append((k, a["cap_min"] >= worst + 1.0 - 1e-9 and a["cap_min"] >= tb["body"] + 1.0 - 1e-9, round(worst, 2), a["cap_min"]))
ok("L2 every arm's cap holds each gated item's due() need from its planned start + 1 min (the 116k window never starved)",
   all(c[1] for c in cap_ok), [c for c in cap_ok if not c[1]])
W = {x: M1.copy_estimate(ARMS, sw) for x, sw in M1.COPIES.items()}
NEED = {x: M1.copy_need_min(ARMS, sw) for x, sw in M1.COPIES.items()}
ok("L2 walks: B 142 min (Q 37, IDX 24, S28 25, S32 26, LP 24), C 146 min (Q 48, G1 22, G2 22, K4 24, DYN 24), every arm runs",
   (W["b"]["runtime_min"], W["c"]["runtime_min"]) == (142.0, 146.0) and all(w["all_run"] for w in W.values())
   and [x["est_min"] for x in W["b"]["walk"]] == [37, 24, 25, 26, 24] and [x["est_min"] for x in W["c"]["walk"]] == [48, 22, 22, 24, 24],
   {x: (w["runtime_min"], [y["est_min"] for y in w["walk"]]) for x, w in W.items()})
ok(f"L2 runtime <= {M1.COPY_RUNTIME_MAX}; the hard stop {M1.BUDGET_MIN} = max(runtime, full-cap end) + 5 rounded up to 5; deadman + 5",
   all(w["runtime_min"] <= M1.COPY_RUNTIME_MAX for w in W.values()) and NEED == {"b": 145.8, "c": 149.8}
   and M1.BUDGET_MIN == 5 * -(-(max(max(w["runtime_min"] for w in W.values()), max(NEED.values())) + M1.START_RESERVE_MIN) // 5) == 155
   and M1.DEADMAN_MIN == 160, (NEED, M1.BUDGET_MIN))

# ---------------------------------------------------------------- L3 the built body / export / facts
BODY = BUILT / "nbm1l_cell7_body.py"
if not BODY.is_file():
    print("SKIP L3-L6 the built light body is not present (set NBM1L_BUILT_DIR to a --light --test-out folder)")
    print(f"{RES['pass']}/{RES['pass'] + RES['fail']} PASS", flush=True)
    sys.exit(1)
btxt = BODY.read_text(encoding="utf-8")
FX = json.loads((BUILT / "nbm1l_build_facts.json").read_text(encoding="utf-8"))
if (BUILT / "nbm1l_export_cells.json").is_file():   # the units kit: the export's cells (CELL 7 stub omitted: size), sha-checked
    _cells = json.loads((BUILT / "nbm1l_export_cells.json").read_text(encoding="utf-8"))
    ok("L3 the kit's export cells = the build's cells (sha per cell; only the CELL 7 stub omitted)",
       len(_cells) == len(FX["nbdf_code_cell_sha256"]) and sum(c is None for c in _cells) == 1
       and all(c is None or hashlib.sha256(c.encode()).hexdigest() == s for c, s in zip(_cells, FX["nbdf_code_cell_sha256"]))
       and hashlib.sha256(btxt.encode()).hexdigest() == FX["body_file_sha256"])
    etxt = "\n\n".join(c if c is not None else "# CELL 7 stub omitted in the units kit (it execs nbm1l_cell7_body.py)" for c in _cells)
else:
    etxt = (BUILT / "nbm1l_code_export.py").read_text(encoding="utf-8")
i_ = btxt.index("NBM1L_OVL_SRC = r'''") + len("NBM1L_OVL_SRC = r'''")
emb = btxt[i_:btxt.index("'''\n", i_)]
j_ = btxt.index("NBM1_LIB_SRC = r'''") + len("NBM1_LIB_SRC = r'''")
emb_lib = btxt[j_:btxt.index("'''\n", j_)]
ok("L3 the embedded overlay = nbm1l_overlay.py (sha = facts) and the embedded lib = nbm1_lib.py (sha = facts)",
   emb == OVL_SRC and hashlib.sha256(emb.encode()).hexdigest() == FX["nbm1"]["overlay_sha256"] == FX["nbm1"]["cfg"]["overlay_sha256"]
   and emb_lib == (HERE / "nbm1_lib.py").read_text(encoding="utf-8") and hashlib.sha256(emb_lib.encode()).hexdigest() == FX["nbm1"]["lib_sha256"])
ok("L3 the overlay runs after nbm1_lib and before the arm plan (sha-checked)",
   btxt.index('exec(compile(NBM1_LIB_SRC, "nbm1_lib.py", "exec"), M1.__dict__)') < btxt.index('exec(compile(NBM1L_OVL_SRC, "nbm1l_overlay.py", "exec"), M1.__dict__)')
   < btxt.index("ARMS = M1.m1_arms(N, _nbm1_q0, M1_SWITCH, M1_VOCAB)") and 'NBM1_CFG["overlay_sha256"]' in btxt)
fns = {n.name: ast.get_source_segment(btxt, n) for n in ast.parse(btxt).body if isinstance(n, ast.FunctionDef)}
for s_ in ('if M1.item_on(arm, "oracle"):           # E25 NB-M1L', 'o["oracle_vs"], o["quality_gate"] = None, "NOT_IN_PLAN_NB_M1L"',
           'return gated(o, need=(), expected_rows=None)', 'NBDF["nbm1"]["light"]', 'M1.item_on(arm, "tf")', 'M1.item_on(arm, "long_oracle")',
           'M1.item_on(arm, "prefix")', 'M1.item_on(arm, "profiles")', 'M1.item_on(arm, "prefill_probe")', 'arm.get("m1_no_ladder")',
           'oom = bool(o["oom_in_log"]) and not o["xid_in_log"]', 'if (is_d or is_lv) and o["smoke_gate"].startswith("FAIL")',
           'o["m1_readback"].get("real_failure")', 'secs=float(M1.PROF_SECS)', '(band in ("40k",) + N.LONG_BANDS or arm.get("m1_lever"))'):
    ok(f"L3 body has {s_}", s_ in btxt)
ok("L3 the oracle block (Q's reference + oracle A/A, a lever's oracle vs Q, its quality gate) sits inside the item switch",
   fns["arm_body"].index('if M1.item_on(arm, "oracle"):') < fns["arm_body"].index('CUR["item"] = "oracle_aa"')
   < fns["arm_body"].index('o["quality_gate"] = quality_gate(o, ref)') < fns["arm_body"].index('"NOT_IN_PLAN_NB_M1L"'))
sw_ = [l for l in etxt.splitlines() if l.startswith("NBM1_ON = ")]
ok("L3 the code export has ONE plain NBM1_ON line (CELL 1, default: every arm on) before CELL 1's header", len(sw_) == 1
   and ast.literal_eval(sw_[0][len("NBM1_ON = "):]) == M1.SWITCH_DEFAULT
   and etxt.index(sw_[0]) < etxt.index("# ==================================================================== CELL 1"), sw_)
m_ = re.search(r'BUDGET = int\(os.environ.get\("RC2_BUDGET_SEC"\) or (\d+) \* 60\)', etxt)
md_ = re.search(r"NBDF_DEADMAN_SEC = (\d+)", etxt)
ok("L3 CELL 1 hard stop 155 min, deadman 160 min", m_ and int(m_.group(1)) == M1.BUDGET_MIN == 155 and md_ and int(md_.group(1)) == 160 * 60,
   (m_ and m_.group(1), md_ and md_.group(1)))
ok("L3 the body's admission rate / window length = the time model's (K100 prefill_admit_floor 3000 tok/s, WIN 90 s)",
   'WIN = float(os.environ.get("E23_WIN_SEC") or 90.0)' in etxt and "E23_WIN_SEC" not in btxt)
i_ = etxt.index('K100_CFG = json.loads(r"""') + len('K100_CFG = json.loads(r"""')
k100cfg = json.loads(etxt[i_:etxt.index('"""', i_)])
ok("L3 K100_CFG: q_i8 True, seqs 24, pin 7,933,526,016, the Q ladder, the K100 windows, admit floor 3000",
   k100cfg["q_i8"] is True and k100cfg["seqs"] == 24 and k100cfg["q_pin"] == PIN and k100cfg["q_ladder"] == [7933526016, 7665090560]
   and k100cfg["windows"] == M1.Q_WINDOWS and float(k100cfg["prefill_admit_floor_tok_s"]) == M1.ADMIT_TOK_S)
_wn = {"WIN": 90.0, "BANDS": {"2k": 2048, "8k": 8192, "40k": 40000, "90k": 90000, "116k": 116000, "20k": 20000}, "PREFILL": {"tok_s": 3000.0}}
exec(fns["win_need"], _wn)
_pairs = [(b, n) for b, n, _ in M1.Q_WINDOWS] + [(w[0], w[1]) for a in ARMS.values() for w in a.get("windows_fixed") or []]
ok("L2 the time model's due() need = the body's own win_need (every window of every arm, the two PROF windows)",
   all(abs(_wn["win_need"](b, n) - M1.win_need_sec(b, n)) < 1e-6 for b, n in _pairs)
   and all(abs(_wn["win_need"]("profile_" + b, n) - M1.win_need_sec(b, n, prof=True)) < 1e-6 for b, n in M1.PROF_WINDOWS))
cps = FX.get("copies") or {}
ok("L3 build facts: two light copies, switch line = repr(COPIES[x]), owners, slugs nbm1l-b / nbm1l-c",
   set(cps) == {"b", "c"} and all(cps[x]["switch_line"] == "NBM1_ON = " + repr(M1.COPIES[x]) and cps[x]["owner"] == M1.OWNERS[x]
                                  and cps[x]["slug"] == f"q27-xhigh-e25-nbm1l-{x}" for x in cps), list(cps))
ok("L3 build facts: every kernel source kept on both copies (tf-ref + the e29 payload; klee100, the FP8 PLE overlay, the Merlin env, dflash2v2)",
   all(cps[x]["dataset_sources"] == ["foysalemonshanto/q27-e25-nb0-tf-ref", "foysalemonshanto/q27-e29-final-payload"]
       and len(cps[x]["model_sources"]) == 4 and any("dflash2v2" in m for m in cps[x]["model_sources"])
       and any("klee100" in m for m in cps[x]["model_sources"]) for x in cps), {x: cps[x].get("dataset_sources") for x in cps})
ok("L3 build facts: light plan recorded (no TF / oracle on any lever, Q items, needs)",
   FX["nbm1"]["light"] is True and FX["nbm1"]["q_items"] == M1.Q_ITEMS and FX["nbm1"]["needs_min"] == NEED
   and all(FX["nbm1"]["plan"][k]["m1_items"]["tf"] is False and FX["nbm1"]["plan"][k]["m1_items"]["oracle"] is False for k in M1.LEVERS))


# ---------------------------------------------------------------- L4 arm_body with fakes: which items run
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


def body_run(arm, switch, smoke_gate="INCONCLUSIVE_SMOKE_INCONCLUSIVE_LADDER", m1=None, fake=True):
    log = []
    nspec = {"FINE_GRAINED_RX": "zz-fine", "DRAFT_GROUP_WARN_RX": "zz-group", "LONG_BANDS": ("90k", "116k"), "BLOCK_TOKENS": 4096,
             "ADMISSION_PROBE_ARMS": (), "smoke_gate": lambda *a, **k: smoke_gate, "pos0_label": lambda *a, **k: {"label": "OK"},
             "spec_mismatch": lambda *a, **k: (False, ""), "qdefer_real_failure": lambda *a, **k: (False, "ok"),
             "k100_rel": lambda s, c: s, "contamination_gate": lambda *a: "INCONCLUSIVE_LADDER", "q_aa_clean": lambda *a: False,
             "long_oracle_gate": lambda *a: "PASS", "prefix_gate": lambda *a: "PASS", "engine_block": lambda *a: 4096,
             "spec_decoding_lines": lambda *a: [], "kv_bound": lambda *a: False}
    ns = {"N": Mod("N", log, nspec), "T": Mod("T", log, {}), "P": Mod("P", log, {"parse_log": lambda *a: []}),
          "L": Mod("L", log, {}), "M1": m1 or M1, "json": json, "re": re, "time": time, "traceback": traceback,
          "FAKE": fake, "CUR": {"kv_bytes": PIN}, "LIVE": {}, "KV_KIT_GATE": {}, "Q_AA": {}, "Q_ACC": {}, "Q_TF_AA": {},
          "PREFILL": {"tok_s": 4000.0}, "K100_CFG": k100cfg, "M1_SWITCH": M1.switch_map(switch), "RESP": {"counts": {}, "arm": None},
          "MEM_RX": re.compile(r"Model loading took"), "_watched": _W, "ROWS": {}}

    def _live(o, k, v):
        o[k] = v
    for nm, ret in (("due", True), ("short_gate_failed", False), ("_mem_gate", (True, "ok")), ("quality_gate", "PASS"),
                    ("band_level", lambda o, b: (6 if b == "116k" else 8, {"source": "fake"})), ("boot_facts", lambda s, o, n: o),
                    ("jit_kernels", lambda *a: []), ("smoke", lambda p, a: {"acceptance": {}, "ladder_skipped": bool(a.get("m1_no_ladder"))}),
                    ("run_window", lambda o, s, a, b, n, tag="": f"{b}@{n}{tag}"), ("profile_window", {"status": "DONE"}),
                    ("i8_readback", {"label": "I8_ENGAGED"}), ("k100_boot_gates", {"real_failure": False}),
                    ("win_need", 10.0)):
        ns[nm] = Rec(nm, log, ret)
    ns["_live"] = _live
    free = set()
    for node in ast.walk(ast.parse(fns["arm_body"])):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            free.add(node.id)
    for nm in free:
        if nm not in ns and nm not in dir(__import__("builtins")):
            ns[nm] = Rec(nm, log)
    exec(fns["arm_body"], ns)
    o = ns["arm_body"](arm, "k100_none" if arm["key"] == "Q" else "q", "q" if arm["key"] == "Q" else None)([_Srv()])
    return o, [c[0] for c in log], log


def windows_called(log):
    return [f"{c[1][3]}@{c[1][4]}{c[1][5] if len(c[1]) > 5 else ''}" for c in log if c[0] == "run_window"]


NEVER = ("oracle", "long_oracle", "prefix_test", "tforced", "admission_probes", "prefill_probe_lever")
for x, sw in M1.COPIES.items():
    o, calls, log = body_run(copy.deepcopy(Q), sw)
    prof_calls = [c for c in log if c[0] == "profile_window"]
    ok(f"L4 copy {x.upper()} Q: smoke without the ladder, the prefill probe, warmup, its six windows; no oracle / long oracle / prefix / TF",
       calls.count("smoke") == 1 and o["smoke"]["ladder_skipped"] is True and calls.count("prefill_probe") == 1 and calls.count("warmup") == 1
       and windows_called(log) == ["8k@16#1", "8k@16#2", "20k@24", "40k@8", "90k@8", "116k@6"]
       and not any(c in calls for c in ("oracle", "long_oracle", "prefix_test", "tforced", "admission_probes")), (calls, windows_called(log)))
    ok(f"L4 copy {x.upper()} Q: the only cache resets are the two long-band ones (no oracle A/A reset)",
       calls.count("reset_prefix") == 2, calls.count("reset_prefix"))
    ok(f"L4 copy {x.upper()} Q: no own profiles; PROF's two 30-s windows {'run' if sw['PROF'] else 'off'}",
       [(c[1][1].get("key"), c[1][2], c[1][3], c[2].get("secs")) for c in prof_calls] ==
       ([("PROF", "8k", 16, 30.0), ("PROF", "20k", 24, 30.0)] if sw["PROF"] else []), [(c[1][2], c[1][3], c[2]) for c in prof_calls])
    ok(f"L4 copy {x.upper()} Q: recorded NOT_IN_PLAN (oracle / long oracle / prefix / TF), not a failure",
       o.get("quality_gate") == "NOT_IN_PLAN_NB_M1L" and o.get("oracle_vs") is None and o.get("long_gate") == "NOT_IN_PLAN_NB_M1"
       and o.get("prefix_gate") == "NOT_IN_PLAN_NB_M1" and o.get("tf_plan") == "NOT_IN_PLAN_NB_M1" and "status" not in o
       and {"oracle", "long_oracle", "prefix_cache", "tf_k100"} <= set(o["items_skipped"]), {k: o.get(k) for k in ("quality_gate", "status", "items_skipped")})
    for k in [k_ for k_ in M1.LEVERS if sw[k_]]:
        a = copy.deepcopy(ARMS[k])
        o, calls, log = body_run(a, sw)
        want = [f"{w[0]}@{w[1]}{w[2]}" for w in a["windows_fixed"]]
        ok(f"L4 copy {x.upper()} {k}: facts, smoke (no ladder), warmup, its windows {want}; no oracle, no TF, no prefill probe / profiles",
           calls.count("smoke") == 1 and o["smoke"]["ladder_skipped"] is True and calls.count("warmup") == 1 and windows_called(log) == want
           and not any(c in calls for c in ("oracle", "long_oracle", "prefix_test", "tforced", "prefill_probe", "profile_window"))
           and o.get("quality_gate") == "NOT_IN_PLAN_NB_M1L" and o.get("tf_plan") == "NOT_IN_PLAN_NB_M1" and "status" not in o,
           (calls, windows_called(log), o.get("status")))
o, calls, log = body_run(copy.deepcopy(DEF["Q"]), M1.COPIES["c"])
ok("L4 harness contrast: the DEFAULT NB-M1 Q plan on the same body calls the oracle twice (reference + A/A), the A/A reset, the long "
   "oracle, the prefix test, its two profiles and TF twice (the harness sees removed items when they run)",
   calls.count("oracle") == 2 and calls.count("reset_prefix") == 3 and calls.count("long_oracle") == 1 and calls.count("prefix_test") == 1
   and calls.count("tforced") == 2 and len([c for c in log if c[0] == "profile_window"]) == 4, calls)
o, calls, log = body_run(copy.deepcopy(DEF["IDX"]), M1.COPIES["b"])
ok("L4 harness contrast: the DEFAULT NB-M1 IDX plan calls its oracle vs Q and its TF", calls.count("oracle") == 1 and calls.count("tforced") == 1, calls)
o, calls, log = body_run(copy.deepcopy(ARMS["IDX"]), M1.COPIES["b"], smoke_gate="FAIL_SMOKE_GARBAGE")
ok("L4 a lever's smoke FAIL (hard label) still stops the arm before any window (NO_GO_SMOKE)",
   o.get("status") == "NO_GO_SMOKE" and not windows_called(log) and "warmup" not in calls, (o.get("status"), calls))
o, calls, log = body_run(copy.deepcopy(Q), M1.COPIES["b"], smoke_gate="FAIL_SMOKE_GARBAGE")
ok("L4 Q's smoke FAIL stays recorded (Q's NBDF-K100 rule: Q never stops on the smoke), its windows run",
   "status" not in o and len(windows_called(log)) == 6 and any("smoke FAIL_SMOKE_GARBAGE" in g for g in o.get("recorded_gates") or []))
import types as _ty  # noqa: E402
M1P = _ty.ModuleType("m1_proxy")
M1P.__dict__.update(M1.__dict__)
M1P.readback = lambda *a, **k: {"real_failure": True, "why": "VLLM_GDN_DECODE_FASTPATH in the workers [None, None, None, None]"}
o, calls, log = body_run(copy.deepcopy(ARMS["G1"]), M1.COPIES["c"], m1=M1P, fake=False)
ok("L4 a lever's read-back real failure still stops that arm (NOT_INTEGRATED) before the smoke / windows",
   o.get("status") == "NOT_INTEGRATED" and o.get("not_integrated") is True and "smoke" not in calls and not windows_called(log), (o.get("status"), calls))
M1G = _ty.ModuleType("m1_proxy_ok")
M1G.__dict__.update(M1.__dict__)
M1G.readback = lambda *a, **k: {"real_failure": False, "ok": True, "why": None}
o, calls, log = body_run(copy.deepcopy(ARMS["G1"]), M1.COPIES["c"], m1=M1G, fake=False)
ok("L4 a lever with a clean read-back (real path, not FAKE) runs its smoke and windows", "smoke" in calls
   and len(windows_called(log)) == 3 and "status" not in o and o["m1_readback"]["ok"] is True, calls)

# ---------------------------------------------------------------- L5 arm_boot / nbdf_main with fakes
class _BSrv:
    def __init__(self, log):
        self._log, self.ready_sec, self.argv = log, 1.0, []

    def log(self):
        return self._log


def boot_ns(outcomes, arm, q_i8=True, left_sec=10 * 3600):
    tries, seen, gates = list(outcomes), [], []
    ns = {"N": N, "T": T, "M1": M1, "re": re, "K100_CFG": k100cfg, "Q_I8": {"on": q_i8, "want": True, "dropped": None},
          "NB4T_BUILD_ENV": {}, "NB1_EXTRA": {}, "NB1_ENV": {}, "NB1_KV": {}, "ARM_OF": {}, "CUR": {}, "BOOT": {}, "RESP": {},
          "NBDF_READY_SEC": 1200, "left": lambda: left_sec, "log": lambda *a, **k: None, "tree_check": lambda a: {"ok": True},
          "boot_gate": lambda: {"ok": True}, "shm_names": lambda: [], "shm_cleanup": lambda x: {}, "q_readback": lambda s, a: {},
          "OOM_ERR": re.compile(r"CUDA out of memory|OutOfMemoryError"), "XID": re.compile(r"Xid \d+"),
          "gated": lambda o, need=(), expected_rows=4: (gates.append((tuple(need), expected_rows)), o)[1],
          "dflash_rb": lambda a, t: {}, "arm_watched": lambda b, c, n: b}

    class _Mon:
        def __init__(self, n):
            self.last, self.stop_ev = None, type("E", (), {"set": lambda s: None})()

        def start(self):
            pass
    ns["LoadMonitor"] = _Mon

    def mfs(name, kv, port, kvb):
        seen.append({"name": name, "port": port, "kvb": kvb, "build_env": dict(ns["NB4T_BUILD_ENV"].get(name) or {})})
        return _BSrv(tries[min(len(seen) - 1, len(tries) - 1)][1])
    ns["merlin_fn_server"] = mfs
    ns["with_servers"] = lambda srvs, ready, b: {"status": tries[min(len(seen) - 1, len(tries) - 1)][0]}
    for fn_ in ("_k100_retry", "arm_boot"):
        exec(fns[fn_], ns)
    o = ns["arm_boot"](arm, lambda ss: {})
    return o, seen, gates


for k in ("Q", "IDX", "G1", "K4"):
    o, seen, gates = boot_ns([("DONE", "ok")], copy.deepcopy(ARMS[k]))
    ok(f"L5 arm_boot {k}: gated on the hard-only evidence (no oracle ref, no oracle rows)", gates == [((), None)], gates)
dq = copy.deepcopy(DEF["Q"])
o, seen, gates = boot_ns([("DONE", "ok")], dq)
ok("L5 arm_boot: an arm whose plan has the oracle (the default NB-M1 Q) keeps the oracle evidence gate", gates == [(("k100_none",), 4)], gates)
o, seen, gates = boot_ns([("SERVER_DIED", "torch.OutOfMemoryError: CUDA out of memory (first request after ready)"), ("DONE", "ok")], copy.deepcopy(ARMS["S28"]))
ok("L5 arm_boot: an OOM after ready -> one rung lower (7,665,090,560), port + 1 (unchanged)", [s_["kvb"] for s_ in seen] == [PIN, 7665090560]
   and seen[1]["port"] == seen[0]["port"] + 1, seen)
o, seen, gates = boot_ns([("SERVER_DIED", "CUDA out of memory\nNVRM: Xid 79"), ("DONE", "ok")], copy.deepcopy(ARMS["G2"]))
ok("L5 arm_boot: OOM with an Xid -> no retry (unchanged)", len(seen) == 1, seen)
o, seen, gates = boot_ns([("DONE", "ok")], copy.deepcopy(ARMS["DYN"]))
ok("L5 arm_boot: a lever's head = Q's int8 into Merlin's build() (unchanged)", seen[0]["build_env"] == {"DRAFT_HEAD": "int8"}, seen)


class _Seq:
    def __init__(self):
        self.state, self.ring, self.log, self.error, self.apply_rec, self.revert_rec, self.verify_rec = "applied", True, [], None, {}, {}, {}

    def apply(self):
        return True, "applied"

    def check(self, k):
        return True, "FULLY_PATCHED"

    def before_arm(self, a):
        return True, "FULLY_PATCHED"

    def finish(self):
        self.state = "reverted"


def main_ns(q_rec=None, d_out=None, gate=True, switch=None):
    calls, R_ = [], {"sections": {"A0": {"merge": {"ok": True}}}, "setup": {}}

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
    ns = {"N": N, "P": P, "T": T, "M1": M1, "R": R_, "ARMS": ARMS, "ISO": {}, "NBDF": {}, "FAKE": False, "ROWS": {"rc2_fp8": 1},
          "K100_CFG": k100cfg, "PENDING": [], "step": step, "save": lambda: None, "log": lambda *a, **k: None, "finalize": lambda: None,
          "left": lambda: 10 * 3600, "merlin_install": lambda: {"kv": {"MERLIN_TREE": "/x"}},
          "apply_60490": lambda: {"status": "DONE", "ok": True}, "k100_setup": lambda: {"ok": True}, "sha_gate": lambda t: (True, "ok"),
          "SEAM_API": object(), "SEAM_LOAD": {}, "SEAM_ENVS": {}, "_tree_py": lambda: (_ for _ in ()).throw(RuntimeError("no tree")),
          "PIN": {"kv_bytes": PIN}, "D_PIN": {"kv_bytes": None}, "SW_PIN": {"kv_bytes": None}, "Q_REC": {},
          "DRAFTER": {"ok": True, "status": None, "why": "x"},
          "KV_KIT_GATE": {"passed": gate, "per_k": {}, "why": None if gate else ["closed"]},
          "_AKEYS": ("name", "est_min"), "PSEQ": None, "SSEQ": None, "run_arm": runner, "M1_SWITCH": M1.switch_map(switch)}
    for fn_ in ("_arm", "skip_arm", "_patch_rec", "_seam_rec", "nbdf_main", "_k100_run_pending", "_k100_tap_arm"):
        exec(fns[fn_], ns)
    ns["nbdf_main"](runner=runner, pseq=_Seq(), sseq=_Seq())
    return calls, R_


def status_of(R_, k):
    return next(iter(R_["sections"].get(k, {"": {}}).values())).get("status")


M1.install(N, ARMS)
for x, sw in M1.COPIES.items():
    c_, R_ = main_ns(switch=sw)
    want = [k for k in M1.ARM_KEYS if sw[k]]
    ok(f"L5 nbdf_main light copy {x.upper()}: exactly {want} launch; every other lever SKIPPED_SWITCH_OFF", c_ == want
       and all(status_of(R_, k) == "SKIPPED_SWITCH_OFF" for k in M1.LEVERS if not sw[k]), (c_, want))
c_, R_ = main_ns(switch=dict(M1.COPIES["b"], Q=False))
ok("L5 Q switched off: copy B's levers still launch", c_ == ["IDX", "S28", "S32", "LP"] and status_of(R_, "Q") == "SKIPPED_SWITCH_OFF", c_)
c_, R_ = main_ns(switch=M1.COPIES["c"], d_out={"G1": {"status": "SERVER_DIED"}, "K4": {"status": "NOT_INTEGRATED", "not_integrated": True}})
ok("L5 a failed lever never skips the next (then the K100 deferred Q-defer-off rerun of the died arm)", c_[:5] == ["Q", "G1", "G2", "K4", "DYN"]
   and c_[5:] == ["G1_ISO"], c_)
c_, R_ = main_ns(gate=False, switch=M1.COPIES["b"])
ok("L5 kit gate closed (the closed build) -> only Q launches; the levers SKIPPED_KV_KIT_GATE", c_ == ["Q"]
   and all(status_of(R_, k) == "SKIPPED_KV_KIT_GATE" for k in ("IDX", "S28", "S32", "LP")), c_)

# ---------------------------------------------------------------- L6 hard-only gates without an oracle + the verdict
_eg = {"N": N, "FAKE": False}
exec(fns["evidence_gate"], _eg)
good = {"probes": {"cases": [{"error": None}]}, "sampled": {"tokens": [1, 2], "alive_after": True, "error": None}}
ok("L6 evidence gate without oracle rows: clean probes + sampled -> None; a probe request error / a dead sampled request -> NO_GO",
   _eg["evidence_gate"](good, None) is None
   and _eg["evidence_gate"](dict(good, probes={"cases": [{"error": "HTTP 500"}]}), None) == "NO_GO_SHORT_PROBE"
   and _eg["evidence_gate"](dict(good, sampled={"tokens": [], "alive_after": False}), None) == "NO_GO_SAMPLED_PATH")
ok("L6 the correctness gate with no oracle ref: clean windows -> CLEAN; hard garbage in a window -> NO_GO_GARBAGE_UNDER_LOAD",
   N.oracle_gate_hard(good, (), (), [{"status": "STEADY"}], False) is None
   and N.oracle_gate_hard(good, (), (), [{"status": "FAIL_GARBAGE_UNDER_LOAD"}], False) == "NO_GO_GARBAGE_UNDER_LOAD")
Wq = {"8k@16#1": {"status": "STEADY", "tok_s": 500.0, "running_eq_level": True, "prefill_done_before_window": True, "level": 16},
      "8k@16#2": {"status": "STEADY", "tok_s": 510.0, "running_eq_level": True, "prefill_done_before_window": True, "level": 16},
      "90k@8": {"status": "STEADY", "tok_s": 340.0, "running_eq_level": True, "prefill_done_before_window": True, "level": 8}}
Wx = {"8k@16#1": {"status": "STEADY", "tok_s": 560.0, "running_eq_level": True, "prefill_done_before_window": True, "level": 16},
      "90k@8": {"status": "STEADY", "tok_s": 300.0, "running_eq_level": True, "prefill_done_before_window": True, "level": 8}}
lt = {"status": "DONE", "correctness": "CLEAN", "quality_gate": "NOT_IN_PLAN_NB_M1L", "oracle_vs": None, "long_gate": "NOT_IN_PLAN_NB_M1",
      "prefix_gate": "NOT_IN_PLAN_NB_M1", "smoke_gate": "INCONCLUSIVE_SMOKE_INCONCLUSIVE_LADDER"}
recs = {"Q": dict(lt, steady=Wq), "IDX": dict(lt, steady=Wx), "S28": dict(lt, steady=Wx), "S32": {"status": "SKIPPED_SWITCH_OFF", "why": "x"}}
try:
    v_ = N.verdict(recs, ARMS)
    vs_ = {k: a_["verdict"] for k, a_ in v_["arms"].items()}
except Exception as e:
    v_, vs_ = {}, repr(e)
print("L6 verdicts:", vs_ if isinstance(vs_, dict) else vs_, "| base", (v_.get("base") or {}).get("label"), flush=True)
ok("L6 verdict over light records: no exception; levers without an oracle are INCONCLUSIVE (record only), switched-off recorded",
   isinstance(vs_, dict) and set(vs_) == set(M1.LEVERS) and vs_["IDX"].startswith("INCONCLUSIVE") and vs_["S28"].startswith("INCONCLUSIVE")
   and vs_["S32"].startswith("SKIPPED"), vs_)
try:
    rr = M1.arm_records(T, recs["IDX"], "IDX", recs["Q"], {"90k": 0.037})
    rb_ = (rr["status"], rr["oracle_vs"], rr["tf_gate"], rr["bands_vs_q"]["bands"]["8k"].get("label"), rr["bands_vs_q"]["bands"]["90k"].get("label"))
except Exception as e:
    rb_ = repr(e)
ok("L6 nbm1_records keep the raw numbers vs Q per band (8k GAIN, 90k REGRESSION) with no oracle / TF", rb_ == ("DONE", None, None, "GAIN", "REGRESSION"), rb_)
print(f"{RES['pass']}/{RES['pass'] + RES['fail']} PASS", flush=True)
sys.exit(1 if RES["fail"] else 0)
