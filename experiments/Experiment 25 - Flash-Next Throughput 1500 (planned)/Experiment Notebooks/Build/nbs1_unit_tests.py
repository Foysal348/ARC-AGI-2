"""E25 NB-S1 quick unit tests (local + the Py3.12 CPU kit), on the real nbdf_lib / nb4t_lib / nb4_ppack and the BUILT body:
  S1  the arm plan: B = the K100 D3s arm under key Q (no draft head, SW-1024, BF16 drafter KV, FLASH_ATTN), G1 / G2 / EH /
      F8 / DYN = B + one change (check_arms); nbdf_lib.launch_check passes every arm's planned launch and refuses a GDN /
      embed-host variable or a draft head on an arm that does not plan it; DYN captures cover every decode batch of both tiers
  S2  install(): nbdf_lib's arm-set constants rebound (ORDER, D_ARMS, SW arms, KV_RUN_OF, K_OF)
  S3  the read-backs: each arm's own change (real failures) and its absence elsewhere; B's prerequisites; the profile plan
  S4  the built body: nbdf_main (fakes) launches B then G1, G2, EH, F8, DYN; B failing never skips an arm; B gets the D-arm
      prerequisites (drafter / kit gate / seam); a failed arm never skips the next; the NB-S1 edits are present
  S5  nbdf_lib.verdict over the NB-S1 arms (B = the base)
Run: python nbs1_unit_tests.py   -> 'FAIL <name>' lines, then '<pass>/<total> PASS'."""
import ast
import copy
import json
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
import nbs1_lib as S1  # noqa: E402

RES = {"pass": 0, "fail": 0}


def ok(name, cond, info=""):
    RES["pass" if cond else "fail"] += 1
    if not cond:
        print("FAIL", name, str(info)[:400], flush=True)


QCAPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96]
N.configure(SEQS=24, CONC_LADDER=(2, 8, 23), Q_CAPTURES=QCAPS, Q_TAIL=["--compilation-config", N.compilation_json(QCAPS)],
            WKEYS=("8k@16#1", "8k@16#2", "profile_20k@24", "40k@16"), PREFIX_RECORD_ONLY=True, SHORT_EXIT_ARMS=(),
            LONG_THR_FLOOR={"90k": 0.037, "116k": 0.054}, TF_REQUIRED=True)
Q_EXTRA = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}", "--max-num-seqs", "24"]
ALL = N.arm_plan(Q_EXTRA, {"VLLM_SERVER_DEV_MODE": "1", "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, 7933526016,
                 "/kaggle/input/models/foysalemonshanto/" + N.DRAFTER_REL, q_env=P.server_env("Q-defer"))
N.restrict_arms(["Q", "D3s", "D4s"])
K100 = {k: ALL[k] for k in N.ORDER}
D3S = copy.deepcopy(K100["D3s"])
PIN = 7933526016


def argv_of(a):
    return ["serve", "x"] + a["extra"] + ["--kv-cache-memory-bytes", str(PIN)]


def env_of(a, **add):
    return dict(a["env"], VLLM_PP_LAYER_PARTITION="25,23", **add)


# ---------------------------------------------------------------- S1 the arm plan
ARMS = S1.s1_arms(N, K100["D3s"], K100["Q"])
ok("S1 arms in order B(Q) G1 G2 EH F8 DYN", tuple(ARMS) == S1.ARM_KEYS, tuple(ARMS))
B = ARMS["Q"]
ok("S1 B = the K100 D3s arm (spec, captures, argv, env) named nbs1_b, A/A on",
   B["spec"] == D3S["spec"] and B["captures"] == D3S["captures"] and B["extra"] == D3S["extra"] and B["env"] == D3S["env"]
   and B["name"] == "nbs1_b" and B["long_aa"] is True and B["method"] == "dflash" and B["k"] == 3, B["name"])
ok("S1 B: SW-1024, BF16 drafter KV (auto), FLASH_ATTN, no draft head (S4)",
   B["env"].get("DFLASH_DRAFT_SW_BLOCK") == "1024" and B["spec"]["kv_cache_dtype"] == "auto" and B["spec"]["attention_backend"] == "FLASH_ATTN"
   and "draft_lm_head_quant" not in B["spec"] and not B["env"].get("DRAFT_HEAD") and B["expect_cfg"]["draft_head"] is None, B["spec"])
ok("S1 the K100 D3s arm is untouched by s1_arms (deep copies)", K100["D3s"] == D3S)
try:
    S1.check_arms(N, T, ARMS)
    ca = True
except ValueError as e:
    ca = repr(e)
ok("S1 check_arms: each arm = B + exactly its change", ca is True, ca)
ok("S1 G1 / G2 env", ARMS["G1"]["env"]["VLLM_GDN_DECODE_FASTPATH"] == "1" and "VLLM_GDN_SHARED_ALIGN_INDEX" not in ARMS["G1"]["env"]
   and ARMS["G2"]["env"]["VLLM_GDN_SHARED_ALIGN_INDEX"] == "1" and "VLLM_GDN_DECODE_FASTPATH" not in ARMS["G2"]["env"])
ok("S1 EH: DFLASH_DRAFT_EMBED_HOST=1 at the same pin", ARMS["EH"]["env"]["DFLASH_DRAFT_EMBED_HOST"] == "1"
   and ARMS["EH"]["kv_bytes"] == B["kv_bytes"] == PIN)
f8 = ARMS["F8"]
ok("S1 F8: fp8_e4m3 drafter KV on TRITON_ATTN + SW 2048 (argv spec and expect_cfg)",
   T.spec_from_argv(f8["extra"]) == f8["spec"] and f8["spec"]["kv_cache_dtype"] == "fp8_e4m3" and f8["spec"]["attention_backend"] == "TRITON_ATTN"
   and f8["env"]["DFLASH_DRAFT_SW_BLOCK"] == "2048" and f8["expect_cfg"]["attention_backend"] == "TRITON_ATTN", f8["spec"])
dy = ARMS["DYN"]
ok("S1 DYN: K 4 + schedule [[1,8,4],[9,24,3]] in the argv spec; k 4", T.spec_from_argv(dy["extra"]) == dy["spec"]
   and dy["spec"]["num_speculative_tokens"] == 4 and dy["spec"]["num_speculative_tokens_per_batch_size"] == [[1, 8, 4], [9, 24, 3]]
   and dy["k"] == 4, dy["spec"])
caps = dy["captures"]
ok("S1 DYN captures: b x 5 (b <= 8) and b x 4 (b 9-24), in the argv compilation config",
   all(b * 5 in caps for b in range(1, 9)) and all(b * 4 in caps for b in range(9, 25)) and N.captures_from_argv(dy["extra"]) == caps, caps)
bad_arms = copy.deepcopy(ARMS)
bad_arms["G1"]["env"]["DFLASH_DRAFT_EMBED_HOST"] = "1"
try:
    S1.check_arms(N, T, bad_arms)
    r_ = "no raise"
except ValueError as e:
    r_ = str(e)
ok("S1 check_arms refuses a second change", "env diff" in r_, r_)

# ---------------------------------------------------------------- S2 install
S1.install(N, ARMS)
ok("S2 install: ORDER / D_ARMS / SW arms / KV_RUN_OF / K_OF", N.ORDER == S1.ARM_KEYS and N.D_ARMS == S1.ARM_KEYS[1:]
   and set(N.SW_BLOCK_ARMS) == set(S1.ARM_KEYS) and set(N.KV_RUN_OF) == set(S1.ARM_KEYS) and N.K_OF["DYN"] == 4 and N.K_OF["EH"] == 3
   and N.SHORT_ARMS == () and N.CONV_TORCH_ARMS == ())
S1.install(N, ARMS)
ok("S2 install is idempotent", N.ORDER == S1.ARM_KEYS)

# launch checks need install() (B and every arm are SW-block arms: nbdf_lib refuses the SW env otherwise)
for k, a in ARMS.items():
    bad = N.launch_check(a, env_of(a), argv_of(a), PIN)
    ok(f"S1 launch_check passes {k}'s planned launch", bad == [], bad)
for k, var in (("Q", "VLLM_GDN_DECODE_FASTPATH"), ("G2", "VLLM_GDN_DECODE_FASTPATH"), ("G1", "VLLM_GDN_SHARED_ALIGN_INDEX"),
               ("Q", "DFLASH_DRAFT_EMBED_HOST")):
    a = ARMS[k]
    bad = N.launch_check(a, env_of(a, **{var: "1"}), argv_of(a), PIN)
    ok(f"S1 launch_check refuses {var} on {k}", any(var in x for x in bad), bad)
a = ARMS["G1"]
bad = N.launch_check(a, env_of(a, DRAFT_HEAD="int8"), argv_of(a), PIN)
ok("S1 S4: DRAFT_HEAD on a NB-S1 launch is refused", any("S4" in x for x in bad), bad)


# ---------------------------------------------------------------- S3 read-backs
def env_rb(**vals):
    return {"workers": [{"rank": r, "values": dict(vals)} for r in range(4)]}


G = "VLLM_GDN_DECODE_FASTPATH"
ok("S3 G1 with the switch on 4 workers -> ok", S1.readback(ARMS["G1"], "", env_rb(**{G: "1"}), {})["ok"] is True)
ok("S3 G1 without the switch -> real failure", S1.readback(ARMS["G1"], "", env_rb(), {})["real_failure"] is True)
ok("S3 G1 with 3 workers -> real failure", S1.readback(ARMS["G1"], "", {"workers": env_rb(**{G: "1"})["workers"][:3]}, {})["real_failure"] is True)
ok("S3 B with a GDN switch -> real failure", S1.readback(B, "", env_rb(**{G: "1"}), {})["real_failure"] is True)
ok("S3 B clean -> ok", S1.readback(B, "", env_rb(), {"checks": {}})["ok"] is True)
ok("S3 EH: embed_host_as_planned True -> ok / False -> real failure",
   S1.readback(ARMS["EH"], "", env_rb(), {"checks": {"embed_host_as_planned": True}})["ok"] is True
   and S1.readback(ARMS["EH"], "", env_rb(), {"checks": {"embed_host_as_planned": False}})["real_failure"] is True)
ok("S3 F8: TRITON_ATTN + block 2048 applied -> ok; either missing -> real failure",
   S1.readback(f8, "", env_rb(), {"checks": {"draft_backend": True, "draft_sw_block_applied": True}})["ok"] is True
   and S1.readback(f8, "", env_rb(), {"checks": {"draft_backend": False, "draft_sw_block_applied": True}})["real_failure"] is True
   and S1.readback(f8, "", env_rb(), {"checks": {"draft_backend": True, "draft_sw_block_applied": False}})["real_failure"] is True)
LOG_OVR = ("(EngineCore pid=1) WARNING 11-01 [vllm.py:1078] Dynamic speculative decoding changes the target verification length at "
           "runtime. Overriding cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE for reliability.")
ok("S3 DYN: cudagraph overridden to PIECEWISE -> real failure; clean -> ok",
   S1.readback(dy, LOG_OVR, env_rb(), {"checks": {}})["real_failure"] is True and S1.readback(dy, "INFO all good", env_rb(), {"checks": {}})["ok"] is True)
ok("S3 DYN: the schedule disabled -> real failure",
   S1.readback(dy, "WARNING Disabling num_speculative_tokens_per_batch_size and falling back", env_rb(), {})["real_failure"] is True)
ok("S3 base_gate: drafter / kit gate / seam each stop B; all ok -> launch",
   S1.base_gate(True, {"ok": False, "status": "DRAFTER_PIN"}, {"passed": True}, True, "")[0] == "NOT_INTEGRATED"
   and S1.base_gate(True, {"ok": True}, {"passed": False}, True, "")[0] == "SKIPPED_KV_KIT_GATE"
   and S1.base_gate(True, {"ok": True}, {"passed": True}, False, "NOT_INTEGRATED: seam")[0] == "NOT_INTEGRATED"
   and S1.base_gate(True, {"ok": True}, {"passed": True}, True, "") is None and S1.base_gate(False, {}, {}, False, "") is None)
ok("S3 profile on B only, 20k@24", S1.profile_plan(B, {"profile": ["20k", 24]}) == [("20k", 24)] and S1.profile_plan(dy, {"profile": ["20k", 24]}) == [])

# ---------------------------------------------------------------- S4 the built body
BODY = HERE / "nbs1_cell7_body.py"
if BODY.is_file():
    btxt = BODY.read_text(encoding="utf-8")
    etxt = (HERE / "nbs1_code_export.py").read_text(encoding="utf-8")
    fns = {n.name: ast.get_source_segment(btxt, n) for n in ast.parse(btxt).body if isinstance(n, ast.FunctionDef)}
    for s_ in ('S1.s1_arms(N, ARMS["D3s"], ARMS["Q"])', "S1.install(N, ARMS)", "S1.check_arms(N, T, ARMS)", "S1.readback(arm, t,",
               "S1.base_gate(q_tree_ok, DRAFTER, KV_KIT_GATE, seam_ok, seam_why)", "S1.profile_plan(arm, NBS1_CFG)",
               '",".join(S1.S1_ENV_KEYS)', "if is_q and not is_d:", "S1.arm_records(", 'secs=float(NBS1_CFG["profile_secs"])'):
        ok(f"S4 body has {s_}", s_ in btxt)
    i_ = etxt.index('K100_CFG = json.loads(r"""') + len('K100_CFG = json.loads(r"""')
    k100cfg = json.loads(etxt[i_:etxt.index('"""', i_)])
    ok("S4 K100_CFG in the notebook (CELL 2b): q_i8 False (S4), wkeys profile_20k@24, seqs 24, pin 7,933,526,016",
       k100cfg["q_i8"] is False and k100cfg["wkeys"][2] == "profile_20k@24" and k100cfg["seqs"] == 24 and k100cfg["q_pin"] == PIN, k100cfg.get("q_i8"))

    class _Seq:
        def __init__(self, apply_ok=True):
            self.state, self.ring, self.log, self.error, self.apply_rec, self.revert_rec = "applied", True, [], None, {}, {}
            self.verify_rec, self.apply_ok = {}, apply_ok

        def apply(self):
            return (True, "applied") if self.apply_ok else (False, "NOT_INTEGRATED: seam apply failed")

        def check(self, k):
            return (True, "FULLY_PATCHED") if self.apply_ok else (False, "NOT_INTEGRATED: seam")

        def before_arm(self, a):
            return True, "FULLY_PATCHED"

        def finish(self):
            self.state = "reverted"

    def main_ns(b_rec=None, d_out=None, drafter_ok=True, gate=True, seam_ok=True, switch=None):
        calls, R_ = [], {"sections": {"A0": {"merge": {"ok": True}}}, "setup": {}}

        def runner(k):
            calls.append(k)
            if k == "Q":
                return dict(b_rec or {"status": "DONE", "pin": PIN})
            return dict((d_out or {}).get(k) or {"status": "DONE", "pin": PIN, "smoke_gate": "PASS"})

        def step(sec, name, est, fn):
            r_ = dict(fn() or {})
            if r_.get("status") in (None, "NOT_RUN"):
                r_["status"] = "DONE"
            R_["sections"].setdefault(sec, {})[name] = r_
            return r_
        ns = {"N": N, "P": P, "T": T, "S1": S1, "R": R_, "ARMS": ARMS, "ISO": {}, "NBDF": {}, "FAKE": False, "ROWS": {"rc2_fp8": 1},
              "K100_CFG": k100cfg, "PENDING": [], "step": step, "save": lambda: None, "log": lambda *a, **k: None, "finalize": lambda: None,
              "left": lambda: 10 * 3600, "merlin_install": lambda: {"kv": {"MERLIN_TREE": "/x"}},
              "apply_60490": lambda: {"status": "DONE", "ok": True}, "k100_setup": lambda: {"ok": True}, "sha_gate": lambda t: (True, "ok"),
              "SEAM_API": object(), "SEAM_LOAD": {}, "SEAM_ENVS": {}, "_tree_py": lambda: (_ for _ in ()).throw(RuntimeError("no tree")),
              "PIN": {"kv_bytes": PIN}, "D_PIN": {"kv_bytes": None}, "SW_PIN": {"kv_bytes": None}, "Q_REC": {},
              "DRAFTER": {"ok": drafter_ok, "status": None if drafter_ok else "DRAFTER_PIN", "why": "x"},
              "KV_KIT_GATE": {"passed": gate, "per_k": {k: True for k in ARMS}, "why": None if gate else ["closed"]},
              "_AKEYS": ("name", "est_min"), "PSEQ": None, "SSEQ": None, "run_arm": runner,
              "NBS1_SWITCH": dict({k: True for k in ARMS}, **(switch or {}))}
        for fn_ in ("_arm", "skip_arm", "_patch_rec", "_seam_rec", "nbdf_main", "_k100_run_pending", "_k100_tap_arm"):
            exec(fns[fn_], ns)
        ns["nbdf_main"](runner=runner, pseq=_Seq(), sseq=_Seq(seam_ok))
        return calls, R_, ns

    def status_of(R_, k):
        return next(iter(R_["sections"].get(k, {"": {}}).values())).get("status")

    c_, R_, ns_ = main_ns()
    ok("S4 nbdf_main: B then G1, G2, EH, F8, DYN launch", c_ == list(S1.ARM_KEYS), c_)
    for nm_, b_ in (("LAUNCH_FAILED", {"status": "LAUNCH_FAILED"}), ("NO_GO_SMOKE", {"status": "NO_GO_SMOKE", "smoke_gate": "FAIL_SMOKE_GARBAGE"}),
                    ("NOT_INTEGRATED", {"status": "NOT_INTEGRATED", "not_integrated": True})):
        c_, R_, ns_ = main_ns(b_rec=b_)
        ok(f"S4 B {nm_} -> every arm still launches (no cross-arm skip)", c_ == list(S1.ARM_KEYS), c_)
    c_, R_, ns_ = main_ns(d_out={"G1": {"status": "SERVER_DIED"}, "EH": {"status": "NOT_INTEGRATED", "not_integrated": True}})
    ok("S4 a failed arm never skips the next (then NBDF's deferred Q-defer-off isolation rerun of the died arm: G1_ISO)",
       c_[:6] == list(S1.ARM_KEYS) and c_[6:] == ["G1_ISO"], c_)
    c_, R_, ns_ = main_ns(drafter_ok=False)
    ok("S4 no drafter -> nothing launches; B and every arm NOT_INTEGRATED (each by its own check)", c_ == []
       and all(status_of(R_, k) == "NOT_INTEGRATED" for k in S1.ARM_KEYS), {k: status_of(R_, k) for k in S1.ARM_KEYS})
    c_, R_, ns_ = main_ns(gate=False)
    ok("S4 kit gate closed (the closed build) -> nothing launches, B SKIPPED_KV_KIT_GATE", c_ == [] and status_of(R_, "Q") == "SKIPPED_KV_KIT_GATE",
       status_of(R_, "Q"))
    c_, R_, ns_ = main_ns(seam_ok=False)
    ok("S4 seam apply failed -> nothing launches, B NOT_INTEGRATED", c_ == [] and status_of(R_, "Q") == "NOT_INTEGRATED", c_)
    # E25 NB-S1 arm switches (CELL 1 NBS1_ON): off -> not booted, SKIPPED_SWITCH_OFF; the other arms unaffected
    off = {"G1": False, "G2": False, "EH": False, "DYN": False}
    c_, R_, ns_ = main_ns(switch=off)
    ok("S4 switch B + F8: only B and F8 launch; G1 / G2 / EH / DYN SKIPPED_SWITCH_OFF", c_ == ["Q", "F8"]
       and all(status_of(R_, k) == "SKIPPED_SWITCH_OFF" for k in off), (c_, {k: status_of(R_, k) for k in S1.ARM_KEYS}))
    c_, R_, ns_ = main_ns(switch=dict(off, F8=False))
    ok("S4 switch B only: only B launches", c_ == ["Q"] and status_of(R_, "F8") == "SKIPPED_SWITCH_OFF", c_)
    c_, R_, ns_ = main_ns(switch={"Q": False})
    ok("S4 switch B off: B SKIPPED_SWITCH_OFF, every lever still launches", c_ == list(S1.ARM_KEYS[1:])
       and status_of(R_, "Q") == "SKIPPED_SWITCH_OFF", c_)
    c_, R_, ns_ = main_ns(switch=off, d_out={"F8": {"status": "SERVER_DIED"}})
    ok("S4 switch B + F8, F8 died: no switched-off arm is started by the rerun path", all(x in ("Q", "F8", "F8_ISO") for x in c_), c_)
    sw_ = next((l for l in etxt.splitlines() if l.startswith("NBS1_ON = ")), "")   # the export = the notebook's code cells
    ok("S4 notebook CELL 1 has the plain NBS1_ON switch line (default B + F8)",
       ast.literal_eval(sw_[len("NBS1_ON = "):] or "None") == {"Q": True, "F8": True, "G1": False, "G2": False, "EH": False, "DYN": False}, sw_)
    ok("S4 OOM at any stage (incl. the first request after ready) steps the D ladder (GPU v1 fix)",
       'oom = bool(o["oom_in_log"]) and not o["xid_in_log"]' in btxt and 'oom = o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]' not in btxt)
    ok("S4 body reads the switch (NBS1_SWITCH from NBS1_ON)", 'globals().get("NBS1_ON")' in btxt and "SKIPPED_SWITCH_OFF" in btxt)
    recs_sw = {"Q": {"status": "DONE"}, "F8": {"status": "DONE"}, "G1": {"status": "SKIPPED_SWITCH_OFF", "why": "x"}}
    try:
        v_ = N.verdict(recs_sw, ARMS)
        vs_ = {k: a["verdict"] for k, a in v_["arms"].items()}
    except Exception as e:
        vs_ = repr(e)
    ok("S5 verdict with switched-off arms: no exception", isinstance(vs_, dict), vs_)
else:
    print("SKIP S4 the built body is not present")

# ---------------------------------------------------------------- S5 verdict on B + arms
recs = {"Q": {"status": "DONE"}, "G1": {"status": "SKIPPED_BUDGET", "why": "x"}, "DYN": {"status": "NOT_INTEGRATED", "not_integrated": True}}
try:
    v = N.verdict(recs, ARMS)
    vv = {k: a["verdict"] for k, a in v["arms"].items()}
except Exception as e:
    vv = repr(e)
ok("S5 verdict over the NB-S1 arms (B = base): skipped / not integrated / not run recorded, no exception",
   isinstance(vv, dict) and set(vv) == set(S1.ARM_KEYS[1:]) and vv["G1"] == "SKIPPED_BUDGET" and vv["DYN"] == "NOT_INTEGRATED", vv)
ok("S5 records: B is reported as B", S1.arm_records({"status": "DONE"}, "Q")["arm"] == "B")
print(f"{RES['pass']}/{RES['pass'] + RES['fail']} PASS", flush=True)
sys.exit(1 if RES["fail"] else 0)
