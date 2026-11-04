"""E27 NBDF-K100 quick unit tests (local + the Py3.12 CPU kit): the K100 variant's own pieces on top of nbdf_unit_tests.py.
  K1  the klee100 method-map gate WITHOUT MTP (k100load_lib.method_map_check, real): a D arm (need_mtp False) with the 48
      routed layers and no MTP / drafter expert rows -> GPU_METHOD_MAP_PASS; Q (need_mtp True) without MTP rows -> FAIL;
      Q with the W4 g128 AutoGPTQ MTP rows on PP1 -> PASS; a dense drafter adds no rows
  K2  the static-scales gate without MTP (k100load_lib.scales_check, real): 4 ranks x their stage's QSA layers -> ok; the MTP
      layer's DEFAULT 1.0 line present or absent -> ok either way (recorded only)
  K3  the K100 body (final design): arms Q > D3s > D4s at 24 sequences; nbdf_main: a failed Q or D3s never skips D4s,
      reruns (nometa / iso / tap) deferred to the end, a k100_setup failure boots nothing; the memory gate never steps down
  K5  the research fixtures: the B1 relative gate (a non-deterministic Q A/A never stops a D arm; a clean one keeps the W4
      rule), the TF envelope gate, the B3 rung walk, the S4 guard, the Q I8 launch check, seqs 24 / captures / ladder, the
      default-binding fix, pos0 labels, prefix record-only, the concurrency trend, a TF death at the end
  K4  the K100 cells: CELL 2b compiles, the payload loader is sha-pinned, the worker extension carries nb4s_env AND
      k100_method_map / k100_kv_scales, the lab is the dataset file, CELL 0 = E18 R2
  K6  the Q failure matrix (boot crash, I8 retry fail, smoke fail, oracle NO_GO, not a usable base, Q-defer real failure,
      Q's ppack before_arm fail): D3s and D4s are still launched (runner called) when their own prerequisites hold
  K7  draft_load_check: dflash_seam's own emit code (DFL_PATCH, verbatim) on 2 PP1 workers parses to PASS (no prefix, a
      vLLM prefix, a re-print); one worker / skipped / PP0-tagged stays a real failure
Run: python nbdf_k100_unit_tests.py   -> 'FAIL <name>' lines, then '<pass>/<total> PASS'."""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
for d in (HERE, E25B, HERE / "e25"):
    if d.is_dir():
        sys.path.insert(0, str(d))
import nbdf_lib as N  # noqa: E402
import nb4t_lib as T  # noqa: E402
import nb4_ppack as P  # noqa: E402
import k100load_lib as KL  # noqa: E402

RES = {"pass": 0, "fail": 0}


def ok(name, cond, info=""):
    if cond:
        RES["pass"] += 1
    else:
        RES["fail"] += 1
        print("FAIL", name, str(info)[:400], flush=True)


# ---------------------------------------------------------------- K1 method map without MTP
EXP = KL.k100_expected_map()
STL = KL.stage_layers((25, 23))


def rows(mtp_rows=False):
    out = []
    for r in range(4):
        st = r // 2
        ex = [{"name": f"model.layers.{L}.mlp.experts", "method": "HummingMoEMethod", "bits": EXP[L][0], "group": EXP[L][1],
               "layer": L, "mtp": False, "source": "method.weight_schema"} for L in sorted(STL[st])]
        if mtp_rows and st == 1:
            ex.append({"name": "mtp.layers.48.mlp.experts", "method": "AutoGPTQMoEMethod", "bits": 4, "group": 128, "layer": 48,
                       "mtp": True, "source": "method.quant_config"})
        out.append({"rank": r, "experts": ex, "errors": [], "drafter_path": "model_runner.speculator.model" if st == 1 else "none"})
    return out


m_d = KL.method_map_check(rows(False), need_mtp=False, partition=(25, 23))
ok("K1 D arm (need_mtp False, no MTP rows) -> GPU_METHOD_MAP_PASS", m_d["gate_ok"] is True and m_d["verdict"] == "GPU_METHOD_MAP_PASS", m_d.get("why"))
m_q0 = KL.method_map_check(rows(False), need_mtp=True, partition=(25, 23))
ok("K1 Q (need_mtp True) without MTP rows -> FAIL (MTP experts missing)", m_q0["gate_ok"] is False and "MTP experts missing" in (m_q0.get("why") or ""), m_q0.get("why"))
m_q = KL.method_map_check(rows(True), need_mtp=True, partition=(25, 23))
ok("K1 Q with the W4 g128 AutoGPTQ MTP rows on PP1 -> PASS", m_q["gate_ok"] is True, m_q.get("why"))
m_x = KL.method_map_check(rows(True), need_mtp=False, partition=(25, 23))
ok("K1 MTP rows on an arm that does not need them -> still PASS (never a false fail)", m_x["gate_ok"] is True, m_x.get("why"))

# ---------------------------------------------------------------- K2 static scales without MTP
_sc = [p for p in (Path(r"D:/ARC AGI 2/_scratch/push_e25/ds_e25_nb0_tf_ref/qsa_kv_scales_262k.json"), HERE / "e25" / "qsa_kv_scales_262k.json")
       if p.is_file()]
SIDE = json.loads(_sc[0].read_text()) if _sc else None
if SIDE:
    want = {KL._layer_of(n): (v["k_scale"], v["v_scale"]) for n, v in SIDE.items()}
    lines = []
    for r in range(4):
        st, tp = r // 2, r % 2
        for L in sorted(x for x in want if x in STL[st]):
            lines.append(f"(Worker_PP{st}_TP{tp} pid={100 + r}) INFO QSA E4M3 KV active on language_model.model.layers.{L}.self_attn.attn: "
                         f"k_scale={want[L][0]} v_scale={want[L][1]}")
    log_no_mtp = "\n".join(lines)
    log_mtp = log_no_mtp + "\n" + "\n".join(f"(Worker_PP1_TP{tp} pid={102 + tp}) INFO QSA E4M3 KV active on mtp.layers.48.self_attn.attn: "
                                            f"k_scale=1.0 v_scale=1.0 (DEFAULT 1.0)" for tp in (0, 1))
    s0, s1 = KL.scales_check(log_no_mtp, SIDE, (25, 23)), KL.scales_check(log_mtp, SIDE, (25, 23))
    ok("K2 scales: no MTP layer line (DFlash2 arm) -> ok", s0["ok"] is True, s0.get("why"))
    ok("K2 scales: the MTP layer at DEFAULT 1.0 (Q) -> ok, recorded", s1["ok"] is True and s1["mtp_default_scale"], s1.get("why"))
    ok("K2 scales: a rank missing -> not ok (real failure kept)", KL.scales_check("\n".join(l for l in lines if "pid=103" not in l), SIDE, (25, 23))["ok"] is False)
else:
    print("SKIP K2 the QSA scales sidecar is not present")

# ---------------------------------------------------------------- K3 the K100 body (final design)
BODY = HERE / "nbdf_k100_cell7_body.py"
NB = HERE.parent / "Q27-XHIGH-E27_NBDF_K100.ipynb"
CFG = {"arms": ["Q", "D3s", "D4s"], "seqs": 24, "conc_ladder": [2, 8, 23], "q_pin": 7933526016, "q_ladder": [7933526016, 7665090560],
       "d_retry": [[7933526016, {"DFLASH_DRAFT_EMBED_HOST": "1"}], [7665090560, {"DFLASH_DRAFT_EMBED_HOST": "1"}],
                   [7396655104, {"DFLASH_DRAFT_EMBED_HOST": "1"}], [6859784192, {"DFLASH_DRAFT_EMBED_HOST": "1"}]],
       "pin_ladder": [7933526016, 7665090560, 7396655104, 6859784192], "q_ref": "k100_none", "q_i8": True,
       "q_captures": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96], "wkeys": ["8k@16#1", "8k@16#2", "profile_8k@16", "40k@16"],
       "windows": [["8k", 16, "#1"], ["8k", 16, "#2"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]],
       "tf_bands": ["2k", "8k", "40k"], "tf_mean_margin": 0.05, "tf_rank1_margin": 0.03, "long_thr_floor": {"90k": 0.037, "116k": 0.054},
       "shorten_arm": "D4s", "shorten_min_left_min": 25, "tap_mode_alt": "pre_combine", "tap_est_min": 15, "tap_cap_min": 10,
       "profile_level": 16, "prefill_admit_floor_tok_s": 3000}
if BODY.is_file():
    btxt = BODY.read_text(encoding="utf-8")
    fns = {n.name: ast.get_source_segment(btxt, n) for n in ast.parse(btxt).body if isinstance(n, ast.FunctionDef)}
    ok("K3 body: the final K100 edits are present", all(x in btxt for x in (
        'N.restrict_arms(K100_CFG["arms"])', "k100_boot_gates(p, s, arm, t)", 'step("A0", "k100_setup", 3, k100_setup)',
        'B3_PIN = int(K100_CFG["q_pin"])', "N.configure(SEQS=int(K100_CFG[\"seqs\"])", '"--max-num-seqs", str(K100_CFG["seqs"])',
        "_k100_retry(arm, is_d, n_oom)", "I8_READBACK_UNVERIFIED", "N.k100_rel(", "N.tf_gate(", 'tforced(p, key="q_tf")',
        "_k100_run_pending(runner)", 'K100_CFG["windows"]', "N.pos0_label(", "_k100_residents(o)")) and "N.pin_plan(" not in btxt, "body")
    ok("K3 body: the memory gate never steps down", fns["_can_step"].rstrip().endswith("return False"), fns["_can_step"])
    ns_r = {"K100_CFG": CFG}
    exec(fns["_k100_retry"] + "\n" + fns["_k100_next_pin"], ns_r)
    walk = [ns_r["_k100_retry"]({}, True, n) for n in range(1, 6)]
    ok("K5 B3 rung walk (D): EMBED_HOST at the SAME pin first, then 7,665,090,560 -> 7,396,655,104 -> 6,859,784,192, then stop",
       walk == [(7933526016, {"DFLASH_DRAFT_EMBED_HOST": "1"}), (7665090560, {"DFLASH_DRAFT_EMBED_HOST": "1"}),
                (7396655104, {"DFLASH_DRAFT_EMBED_HOST": "1"}), (6859784192, {"DFLASH_DRAFT_EMBED_HOST": "1"}), None], walk)
    ok("K5 B3 rung walk (Q): one 0.25-GiB rung, never -1 GiB first", ns_r["_k100_retry"]({}, False, 1) == (7665090560, {})
       and ns_r["_k100_retry"]({}, False, 2) is None)
    ok("K5 _k100_next_pin: 0.25-GiB rungs", ns_r["_k100_next_pin"](7933526016) == 7665090560 and ns_r["_k100_next_pin"](6859784192) is None)

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

    N.configure(SEQS=24, CONC_LADDER=(2, 8, 23), Q_CAPTURES=list(CFG["q_captures"]),
                Q_TAIL=["--compilation-config", N.compilation_json(CFG["q_captures"])], WKEYS=tuple(CFG["wkeys"]),
                PREFIX_RECORD_ONLY=True, SHORT_EXIT_ARMS=(), LONG_THR_FLOOR=dict(CFG["long_thr_floor"]), TF_REQUIRED=True)
    Q_EXTRA = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}", "--max-num-seqs", "24"]
    ARMS_ALL = N.arm_plan(Q_EXTRA, {"VLLM_SERVER_DEV_MODE": "1", "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, 7933526016,
                          "/kaggle/input/models/foysalemonshanto/" + N.DRAFTER_REL, q_env=P.server_env("Q-defer"))
    N.restrict_arms(CFG["arms"])
    ARMS = {k: ARMS_ALL[k] for k in N.ORDER}
    ok("K3 restrict_arms: Q > D3s > D4s (the given order), SW arms D3s / D4s, no short / conv / short-exit arms",
       N.ORDER == ("Q", "D3s", "D4s") and N.D_ARMS == ("D3s", "D4s") and set(N.SW_BLOCK_ARMS) == {"D3s", "D4s"}
       and N.SHORT_ARMS == () and N.CONV_TORCH_ARMS == () and N.SHORT_EXIT_ARMS == () and set(N.KV_RUN_OF) == {"D4s", "D3s"}, N.ORDER)
    try:
        N.check_arm_plan(ARMS)
        _cap_ok = True
    except ValueError as e:
        _cap_ok = repr(e)
    ok("K5 the K100 arm plan passes check_arm_plan (Q's trailing captures, every arm at 24 sequences)", _cap_ok is True, _cap_ok)
    ok("K5 captures at 24: Q = Merlin registry 1..96, D3s 4..96 step 4, D4s 5..120 step 5, every decode batch has a FULL graph",
       ARMS["Q"]["captures"] == CFG["q_captures"] and ARMS["D3s"]["captures"] == list(range(4, 97, 4))
       and ARMS["D4s"]["captures"] == list(range(5, 121, 5)) and all(a["coverage"]["covers_all"] for a in ARMS.values())
       and all(a["expect_cfg"]["max_num_seqs"] == 24 for a in ARMS.values()), {k: a["captures"][-1] for k, a in ARMS.items()})
    ok("K5 the ladder at 24: (2, 8, 23)", N.CONC_LADDER == (2, 8, 23))
    ok("K5 default-binding fix: level_for_band / resident_level read SEQS at call time (24, not 8)",
       N.level_for_band("40k", 40000, 540, 540 / (32 + 12), 4096)[0] > 8 or N.level_for_band("40k", 40000, 540, 540 / (32 + 12), 4096)[0] <= 24,
       N.level_for_band("40k", 40000, 540, 540 / (32 + 12), 4096))
    import inspect as _insp
    ok("K5 no seqs=SEQS default left in nbdf_lib (A's trap at :558)", all(_insp.signature(f_).parameters["seqs"].default is None
       for f_ in (N.level_for_band, N.resident_level, N.plan_level, N.draft_logits_mib)))

    def _qd():
        c = {"defer_park": 24406, "flush_exec": 21000, "flush_empty": 3404, "flush_recv": 0, "flush_consume": 0, "flush_rpc": 2,
             "flush_shutdown": 0, "exec_share": 0.8604, "fallback_share": 0.0, "clean": False}
        return {"label": "ENGAGED_FALLBACK", "detail": {"inits": 4, "expect": 4, "defer_per_rank": {"1": dict(c), "2": dict(c)}}}
    QK = {"status": "DONE", "correctness": "CLEAN", "quality_gate": "INCONCLUSIVE", "oracle_vs": "INCONCLUSIVE_NO_ORACLE", "long_gate": "PASS",
          "q_readback_final": _qd(), "pin": 7933526016, "free_after_ready_min": {"min": [700, 720, 719, 730]}}

    def main_ns(q_rec, setup_ok=True, d_out=None):
        calls, R_ = [], {"sections": {"A0": {"merge": {"ok": True}}}, "setup": {}}

        def runner(k):
            calls.append(k)
            if k == "Q":
                return dict(q_rec)
            return dict((d_out or {}).get(k) or {"status": "DONE", "pin": 7933526016, "smoke_gate": "PASS"})

        def step(sec, name, est, fn):
            r_ = dict(fn() or {})
            if r_.get("status") in (None, "NOT_RUN"):
                r_["status"] = "DONE"
            R_["sections"].setdefault(sec, {})[name] = r_
            return r_
        ns = {"N": N, "P": P, "T": T, "R": R_, "ARMS": ARMS, "ISO": {}, "NBDF": {}, "FAKE": False, "ROWS": {}, "K100_CFG": CFG, "PENDING": [],
              "step": step, "save": lambda: None, "log": lambda *a, **k: None, "finalize": lambda: None, "left": lambda: 4 * 3600,
              "merlin_install": lambda: {"kv": {"MERLIN_TREE": "/x"}}, "apply_60490": lambda: {"status": "DONE", "ok": True},
              "k100_setup": lambda: {"ok": setup_ok, "why": None if setup_ok else "K100_NO_SCALES"},
              "sha_gate": lambda t: (True, "ok"), "SEAM_API": object(), "SEAM_LOAD": {}, "SEAM_ENVS": {},
              "_tree_py": lambda: (_ for _ in ()).throw(RuntimeError("no tree")),
              "PIN": {"kv_bytes": 7933526016}, "D_PIN": {"kv_bytes": None}, "SW_PIN": {"kv_bytes": None}, "Q_REC": {},
              "DRAFTER": {"ok": True}, "KV_KIT_GATE": {"passed": True, "per_k": {"D4s": True, "D3s": True}},
              "_AKEYS": ("name", "est_min"), "PSEQ": None, "SSEQ": None, "run_arm": runner}
        for fn_ in ("_arm", "skip_arm", "_patch_rec", "_seam_rec", "nbdf_main", "_k100_run_pending", "_k100_tap_arm"):
            exec(fns[fn_], ns)
        ns["nbdf_main"](runner=runner, pseq=_Seq(), sseq=_Seq())
        return calls, R_, ns

    c_, R_, ns_ = main_ns(QK)
    ok("K3 nbdf_main: the K100Q-like Q (ENGAGED_FALLBACK, no reference rows) -> Q, D3s, D4s run, D starts at 7,933,526,016",
       c_ == ["Q", "D3s", "D4s"] and ns_["D_PIN"]["kv_bytes"] == 7933526016, (c_, ns_["D_PIN"]))
    c_, R_, ns_ = main_ns(dict(QK, status="SERVER_DIED"))
    ok("K3 nbdf_main: Q failed -> recorded, D3s and D4s still run (no cross-arm skips)", c_ == ["Q", "D3s", "D4s"]
       and "base_not_usable" in ns_["NBDF"], (c_, ns_["NBDF"].get("base_not_usable")))
    c_, R_, ns_ = main_ns(QK, d_out={"D3s": {"status": "NO_GO_SMOKE", "smoke_gate": "FAIL_SMOKE_DFLASH_ZERO_ACCEPT",
                                             "smoke_pos0": {"label": "STRUCTURAL", "pos0": 0.0}, "ingraph_meta": {"kernel_map_refresh": True}}})
    ok("K3 nbdf_main: D3s zero-accept STRUCTURAL -> D4s still runs, THEN the deferred nometa / tap reruns (no iso on zero accept)",
       c_[:3] == ["Q", "D3s", "D4s"] and c_[3:] == ["D3s_NOMETA", "D3s_TAP"], c_)
    c_, R_, ns_ = main_ns(QK, d_out={"D3s": {"status": "SERVER_DIED"}})
    ok("K3 nbdf_main: D3s crashed -> D4s runs, then the one Q-defer-off isolation rerun", c_ == ["Q", "D3s", "D4s", "D3s_ISO"], c_)
    c_, R_, ns_ = main_ns(QK, setup_ok=False)
    ok("K3 nbdf_main: k100_setup failed (no scales) -> no arm boots (a global boot failure), every arm NOT_INTEGRATED", c_ == []
       and all(v.get("status") == "NOT_INTEGRATED" for k in N.ORDER for v in R_["sections"][k].values()), c_)

    # ---------------------------------------------------------------- K5 the B1 relative gate + the TF envelope
    ok("K5 B1 k100_rel: a clean Q A/A keeps FAIL_HIGH_MARGIN (W4 rule); a non-deterministic one records it",
       N.k100_rel("FAIL_HIGH_MARGIN", True) == "FAIL_HIGH_MARGIN" and N.k100_rel("FAIL_HIGH_MARGIN", False) == "NONDET_BASE_RECORDED"
       and N.k100_rel("EXACT", False) == "EXACT" and N.q_aa_clean({"oracle": "EXACT", "ladder": "NEAR_TIE"})
       and not N.q_aa_clean({"oracle": "FAIL_HIGH_MARGIN", "ladder": "EXACT"}) and not N.q_aa_clean({}))
    ladder_bad = {"alone": {"tokens": [1]}, **{str(c): {"vs_alone": {"status": "FAIL_HIGH_MARGIN"}} for c in N.CONC_LADDER}}
    smk = {"tokens": [N.SMOKE_TOKENS] * 2, "acceptance": {"status": "OK", "drafts": 500, "accept_rate": 0.6, "per_pos": [0.8, 0.6, 0.4]},
           "ladder": ladder_bad}
    ok("K5 B1 + B2 smoke: a ladder FAIL_HIGH_MARGIN -> INCONCLUSIVE (recorded, the arm runs), with or without a clean Q A/A",
       N.smoke_gate(smk, False, q_nondet=True) == "INCONCLUSIVE_SMOKE_NONDET_BASE_LADDER"
       and N.smoke_gate(smk, False, q_nondet=False) == "INCONCLUSIVE_SMOKE_CONCURRENCY_CONTAMINATION")
    ok("K5 zero-accept canary stays a real failure on a non-deterministic base",
       N.smoke_gate(dict(smk, acceptance={"status": "OK", "drafts": 500, "accept_rate": 0.0, "per_pos": [0, 0, 0]}), False, q_nondet=True)
       == "FAIL_SMOKE_DFLASH_ZERO_ACCEPT")
    q_aa_tf = {"2k": {"mean_abs_dlogprob": 0.659, "rank1_agreement": 0.856}, "8k": {"mean_abs_dlogprob": 0.098, "rank1_agreement": 0.949},
               "40k": {"mean_abs_dlogprob": 0.283, "rank1_agreement": 0.914}}             # K3_ONLY v2's klee100 A/A
    d_in = {"2k": {"mean_abs_dlogprob": 0.70, "rank1_agreement": 0.84}, "8k": {"mean_abs_dlogprob": 0.12, "rank1_agreement": 0.93},
            "40k": {"mean_abs_dlogprob": 0.30, "rank1_agreement": 0.90}}
    d_out = dict(d_in, **{"8k": {"mean_abs_dlogprob": 0.20, "rank1_agreement": 0.93}})
    ok("K5 TF envelope: D inside Q's A/A (+0.05 / -0.03) -> PASS; 8k mean 0.20 > 0.148 -> FAIL; a missing band -> INCONCLUSIVE",
       N.tf_gate(d_in, q_aa_tf)["status"] == "PASS" and N.tf_gate(d_out, q_aa_tf)["status"] == "FAIL_TF_OUTSIDE_Q_AA"
       and N.tf_gate({"2k": d_in["2k"]}, q_aa_tf)["status"] == "INCONCLUSIVE_TF"
       and N.tf_gate(dict(d_in, **{"40k": {"mean_abs_dlogprob": 0.30, "rank1_agreement": 0.88}}), q_aa_tf)["status"] == "FAIL_TF_OUTSIDE_Q_AA")
    ok("K5 pos0 labels: < 0.05 STRUCTURAL, 0.05-0.5 TAP_SUSPECT, else OK", N.pos0_label({"per_pos": [0.01]})["label"] == "STRUCTURAL"
       and N.pos0_label({"per_pos": [0.3]})["label"] == "TAP_SUSPECT" and N.pos0_label({"per_pos": [0.86]})["label"] == "OK")

    def _w(tok, lvl, al=3.2):
        return {"status": "STEADY", "running_eq_level": True, "preemptions": 0, "tok_s": tok, "tick_ms": 50.0, "level": lvl,
                "mean_acceptance_length": al, "prefix_hit_rate": 0.5}

    def _rec(t8, t20, t40, t90, t116, **kw):
        r = {"status": "DONE", "correctness": "CLEAN", "quality_gate": "PASS", "oracle_vs": "EXACT", "long_gate": "PASS", "smoke_gate": "PASS",
             "prefix_gate": "PASS", "env_readback": {"ok": True}, "config_readback": {"ok": True, "log": {}}, "dflash_readback": {"ok": True},
             "q_readback_final": {"label": "ENGAGED"}, "memory_gate_ready": {"ok": True}, "memory_gate_first_decode": {"ok": True},
             "long_oracle": {"complete": True, "errors": [], "garbage": [], "vs_q_long": {"status": "EXACT"}},
             "steady": {"8k@16#1": _w(t8[0], 16), "8k@16#2": _w(t8[1], 16), "20k@24": _w(t20, 24), "40k@16": _w(t40, 16),
                        "90k@8": _w(t90, 8), "116k@6": _w(t116, 6)},
             "pool": {"max_concurrency": 12.6}, "workers": {"num_gpu_blocks": [480]}, "pin": 7933526016}
        r.update(kw)
        return r
    QR = _rec((550, 555), 670, 530, 320, 260, oracle_aa={"vs_q": {"status": "FAIL_HIGH_MARGIN"}})
    DR = _rec((650, 655), 760, 625, 360, 295, oracle_vs="NONDET_BASE_RECORDED", tf_gate={"status": "PASS"},
              long_oracle={"complete": True, "errors": [], "garbage": [], "vs_q_long": {"status": "NONDET_BASE_RECORDED"}},
              prefix_gate="PREFIX_HIT_ACCEPTANCE_COLLAPSE", smoke={"acceptance": {"status": "OK", "per_pos": [0.86, 0.73, 0.6], "al": 3.2}})
    v = N.verdict({"Q": QR, "D3s": DR}, ARMS)
    a3 = v["arms"]["D3s"]
    ok("K5 B1 fixture: a non-deterministic Q A/A does NOT stop D3s: NONDET token ids + TF PASS + prefix collapse (record-only) -> WIN",
       a3["verdict"] == "WIN" and "PREFIX_HIT_ACCEPTANCE_COLLAPSE" in a3["flags"], (a3.get("verdict"), a3.get("why"), a3.get("flags")))
    tr = (a3.get("k100_records") or {}).get("conc_trend") or {}
    ok("K5 the concurrency trend is reported separately for 8k@16 and 20k@24", tr.get("8k@16") and tr.get("20k@24")
       and tr.get("direction") in ("RISES_WITH_CONCURRENCY", "FALLS_WITH_CONCURRENCY"), tr)
    v2 = N.verdict({"Q": QR, "D3s": dict(DR, tf_gate={"status": "FAIL_TF_OUTSIDE_Q_AA", "why": "8k"})}, ARMS)
    ok("K5 the TF envelope fail -> CORRECTNESS_FAIL_TF (D3s only)", v2["arms"]["D3s"]["verdict"] == "CORRECTNESS_FAIL_TF", v2["arms"]["D3s"].get("verdict"))
    ok("K5 TF PASS + speed PASS -> WIN (no TF_NOT_VERIFIED flag)", a3["verdict"] == "WIN" and "TF_NOT_VERIFIED" not in a3["flags"]
       and not a3.get("tf_verification_pending"), (a3.get("verdict"), a3.get("flags")))
    for nm_, dr_ in (("TF absent", {k: v for k, v in DR.items() if k != "tf_gate"}), ("TF INCONCLUSIVE", dict(DR, tf_gate={"status": "INCONCLUSIVE_TF"}))):
        a_ = N.verdict({"Q": QR, "D3s": dr_}, ARMS)["arms"]["D3s"]
        ok(f"K5 {nm_} -> INCONCLUSIVE_TF, speed_verdict WIN kept, flag TF_NOT_VERIFIED, speed data intact",
           a_["verdict"] == "INCONCLUSIVE_TF" and a_.get("speed_verdict") == "WIN" and a_.get("tf_verification_pending") is True
           and "TF_NOT_VERIFIED" in a_["flags"] and a_.get("speed") == a3.get("speed") and a_.get("long") == a3.get("long"),
           (a_.get("verdict"), a_.get("speed_verdict"), a_.get("flags")))
    N.configure(TF_REQUIRED=False)
    a_ = N.verdict({"Q": QR, "D3s": {k: v for k, v in DR.items() if k != "tf_gate"}}, ARMS)["arms"]["D3s"]
    ok("K5 W4 (TF_REQUIRED off, no TF gate exists there): TF absent keeps the W4 verdict", a_["verdict"] == "WIN" and not a_.get("tf_verification_pending"), a_.get("verdict"))
    N.configure(TF_REQUIRED=True)
    v3 = N.verdict({"Q": QR, "D3s": dict(DR, status="SERVER_DIED", died_in="tf_k100", measured_before_tf=True)}, ARMS)
    ok("K5 a server death in the final TF keeps the measured verdict (flag TF_SERVER_DIED_AT_END)", v3["arms"]["D3s"]["verdict"] == "WIN"
       and "TF_SERVER_DIED_AT_END" in v3["arms"]["D3s"]["flags"], (v3["arms"]["D3s"].get("verdict"), v3["arms"]["D3s"].get("why")))
    ok("K5 the D4s 'c >= 0.80' data gate and the no-hope cut are recorded only", "d4s_data_gate_c_ge_0_80" in (a3.get("k100_records") or {})
       and (a3.get("k100_records") or {}).get("role") == "recorded only")

    # ---------------------------------------------------------------- K5 the S4 guard + the Q I8 launch check
    d3 = ARMS["D3s"]
    argv_d = ["serve", "x"] + d3["extra"] + ["--kv-cache-memory-bytes", "7933526016"]
    env_d = dict(d3["env"], VLLM_PP_LAYER_PARTITION="25,23")
    ok("K5 S4: a clean D3s launch passes", N.launch_check(d3, env_d, argv_d, 7933526016) == [], N.launch_check(d3, env_d, argv_d, 7933526016))
    ok("K5 S4: DRAFT_HEAD on a dflash launch is refused", any("S4" in x for x in N.launch_check(d3, dict(env_d, DRAFT_HEAD="int8"), argv_d, 7933526016)))
    sp_q = dict(N.Q_SPEC, draft_lm_head_quant="int8")
    q = dict(ARMS["Q"], draft_head="int8")
    argv_q = ["serve", "x", "--speculative-config", json.dumps(sp_q), "--compilation-config", N.compilation_json(CFG["q_captures"])] + q["extra"] + \
             ["--kv-cache-memory-bytes", "7933526016"]
    env_q = dict(q["env"], VLLM_PP_LAYER_PARTITION="25,23")
    ok("K5 Q I8: the INT8 head in Merlin's spec passes the launch check only when I8 is on", N.launch_check(q, env_q, argv_q, 7933526016) == []
       and N.launch_check(dict(q, draft_head=None), env_q, argv_q, 7933526016) != [], N.launch_check(q, env_q, argv_q, 7933526016))
    ok("K5 a real fallback (MTP on a D arm, eager drafter, wrong K) is a real failure", N.spec_mismatch(True, {}, {"contradictions": ["the drafter runs EAGER (x)"]})[0]
       and N.spec_mismatch(True, {"contradictions": ["spec_k 3 != 4"]}, {})[0] and not N.spec_mismatch(True, {"contradictions": ["captures [5] != [5, 10]"]}, {})[0])
    if NB.is_file():
        cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
        c2b_src = next((c for c in cells if "CELL 2b — E27 K100" in c), None)
        import textwrap
        c2b = textwrap.dedent(c2b_src[len("if not RERUN:\n"):])
        ok("K4 CELL 2b: compiles, payload sha-pinned, overlay merge + K6 + digests + scales + the I8 read-back present", compile(c2b, "c2b", "exec") is not None
           and 'K100Q_PAYLOAD_SHA = "fd92a814040aea6dda12855badf801675580dcd78b2f2135eb0b86121dd80880"' in c2b and "def merge_ckpt():" in c2b
           and "def k100_scales(" in c2b and "KL.k6_check(CKPT)" in c2b and "def k100_boot_gates(" in c2b and "def k100_setup(" in c2b
           and "def i8_readback(" in c2b)
        cfg_line = next(l for l in c2b.splitlines() if l.startswith("K100_CFG = json.loads("))
        _ns = {"json": json}
        exec(cfg_line, _ns)
        ok("K4 the notebook's K100_CFG = the coordinator's final design", {k: _ns["K100_CFG"].get(k) for k in CFG} == CFG,
           {k: (_ns["K100_CFG"].get(k), CFG[k]) for k in CFG if _ns["K100_CFG"].get(k) != CFG[k]})
        c6 = next(c for c in cells if "NB1_EXT_SRC = " in c)
        ok("K4 worker extension = NB1 + nb4s_env + the klee100 method map / kv scales (sha-checked blob)", "E27 K100: NBDF's extension" in c6)
        c1 = cells[1]
        ok("K4 CELL 1: the lab is the tf-ref dataset file (no packed literal), the klee100 inputs", 'K100_LAB_SHA = "ef393fbb' in c1
           and "K100_DIR = MD /" in c1 and "MERLIN_OK = K100_DIR is not None" in c1)
        e18 = Path(r"D:/ARC AGI 2/experiments/Experiment 18 - S1-60K on vLLM 0.32 (planned)/Experiment Notebooks/Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb")
        if e18.is_file():
            ok("K4 CELL 0 = E18 R2 cell 1 verbatim", cells[0] == "".join(json.loads(e18.read_text(encoding="utf-8"))["cells"][1]["source"]).replace("\r\n", "\n"))
        ok("K4 notebook < 400 kB", NB.stat().st_size < 400_000, NB.stat().st_size)
    else:
        print("NOTE K4 the K100 notebook is not present (the kit bundles the body only)")
else:
    print("NOTE K3 the K100 body file is not present")


# ---------------------------------------------------------------- K6 Q failure matrix: D3s and D4s still launch (coordinator 2026-11-01)
if BODY.is_file():
    class _SeqQFail(_Seq):
        def before_arm(self, a):
            return (False, "ppack apply failed (Q only)") if a.get("key") == "Q" else (True, "FULLY_PATCHED")

    def main_ns2(q_rec, pseq=None):
        calls, R_ = [], {"sections": {"A0": {"merge": {"ok": True}}}, "setup": {}}

        def runner(k):
            calls.append(k)
            return dict(q_rec) if k == "Q" else {"status": "DONE", "pin": 7933526016, "smoke_gate": "PASS"}

        def step(sec, name, est, fn):
            r_ = dict(fn() or {})
            if r_.get("status") in (None, "NOT_RUN"):
                r_["status"] = "DONE"
            R_["sections"].setdefault(sec, {})[name] = r_
            return r_
        ns = {"N": N, "P": P, "T": T, "R": R_, "ARMS": ARMS, "ISO": {}, "NBDF": {}, "FAKE": False, "ROWS": {}, "K100_CFG": CFG, "PENDING": [],
              "step": step, "save": lambda: None, "log": lambda *a, **k: None, "finalize": lambda: None, "left": lambda: 4 * 3600,
              "merlin_install": lambda: {"kv": {"MERLIN_TREE": "/x"}}, "apply_60490": lambda: {"status": "DONE", "ok": True},
              "k100_setup": lambda: {"ok": True}, "sha_gate": lambda t: (True, "ok"), "SEAM_API": object(), "SEAM_LOAD": {}, "SEAM_ENVS": {},
              "_tree_py": lambda: (_ for _ in ()).throw(RuntimeError("no tree")),
              "PIN": {"kv_bytes": 7933526016}, "D_PIN": {"kv_bytes": None}, "SW_PIN": {"kv_bytes": None}, "Q_REC": {},
              "DRAFTER": {"ok": True}, "KV_KIT_GATE": {"passed": True, "per_k": {"D4s": True, "D3s": True}},
              "_AKEYS": ("name", "est_min"), "PSEQ": None, "SSEQ": None, "run_arm": runner}
        for fn_ in ("_arm", "skip_arm", "_patch_rec", "_seam_rec", "nbdf_main", "_k100_run_pending", "_k100_tap_arm"):
            exec(fns[fn_], ns)
        ns["nbdf_main"](runner=runner, pseq=pseq or _Seq(), sseq=_Seq())
        return calls, R_, ns
    QFAIL = {"boot crash (LAUNCH_FAILED)": dict(QK, status="LAUNCH_FAILED", oom_in_log=False),
             "boot crash (SERVER_DIED)": dict(QK, status="SERVER_DIED"),
             "I8 retry failed (I8_READBACK_UNVERIFIED)": dict(QK, status="I8_READBACK_UNVERIFIED"),
             "smoke fail (NO_GO_SMOKE)": dict(QK, status="NO_GO_SMOKE", smoke_gate="FAIL_SMOKE_GARBAGE"),
             "oracle NO_GO (NO_GO_QUALITY)": dict(QK, status="NO_GO_QUALITY", quality_gate="FAIL"),
             "not a usable base (garbage)": dict(QK, correctness="NO_GO_GARBAGE"),
             "not a usable base (Q-defer real failure)": dict(QK, q_readback_final={"label": "NOT_INTEGRATED", "detail": {"inits": 2, "expect": 4}})}
    for nm_, q_ in QFAIL.items():
        c_, R_, ns_ = main_ns2(q_)
        ok(f"K6 Q fails ({nm_}) -> D3s and D4s still launch", c_[:3] == ["Q", "D3s", "D4s"] and "D3s" in c_ and "D4s" in c_, c_)
    c_, R_, ns_ = main_ns2(QK, pseq=_SeqQFail())
    ok("K6 Q's nb4_ppack before_arm fails -> Q NOT_INTEGRATED (not launched), D3s and D4s pass their own checks and launch",
       c_ == ["D3s", "D4s"] and all(v.get("status") == "NOT_INTEGRATED" for v in R_["sections"]["Q"].values()), (c_, R_["sections"].get("Q")))
    ok("K6 the body has no 'no base: Q' skip left", "no base: Q" not in btxt)

# ---------------------------------------------------------------- K7 draft_load_check: dflash_seam's own emit -> the parser
import contextlib
import io
import os as _os
import types as _types
import dflash_seam as DS  # noqa: E402


def _patch_fn(patch_text, name):
    """One function of a dflash_seam unified-diff patch, verbatim (the '+' lines of the hunk, '+' stripped)."""
    lines = [l[1:] for l in patch_text.splitlines() if l.startswith("+") and not l.startswith("+++")]
    i = next(n for n, l in enumerate(lines) if l.startswith(f"def {name}("))
    j = next((n for n in range(i + 1, len(lines)) if lines[n].startswith(("def ", "class ", "_E27_", "@"))), len(lines))
    src = "\n".join(lines[i:j])
    tree = ast.parse(src)
    hit = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(hit) == 1, name
    return ast.get_source_segment(src, hit[0])


_ns7 = {"nn": _types.SimpleNamespace(Module=object)}
exec(_patch_fn(DS.DFL_PATCH, "_e27_log") + "\n\n" + _patch_fn(DS.DFL_PATCH, "_e27_check_loaded"), _ns7)


class _M:
    def named_parameters(self):
        return [(f"model.layers.{i}.w{j}", None) for i in range(5) for j in range(13)] + [("fc.weight", None), ("hidden_norm.weight", None)]


def _emit(pid, skip=False):
    buf = io.StringIO()
    old_pid, old_env = _os.getpid, dict(_os.environ)
    try:
        _os.getpid = lambda: pid
        _os.environ["DFLASH_SEAM"] = "1"
        if skip:
            _os.environ["DFLASH_LOAD_CHECK"] = "0"
        else:
            _os.environ.pop("DFLASH_LOAD_CHECK", None)
        with contextlib.redirect_stdout(buf):
            names = [n for n, _ in _M().named_parameters()]
            _ns7["_e27_check_loaded"](_M(), set(names), {"embed_tokens": 1}, {n: None for n in names})
    finally:
        _os.getpid = old_pid
        _os.environ.clear()
        _os.environ.update(old_env)
    return buf.getvalue().strip()


L1, L2 = _emit(2410), _emit(2411)
ok("K7 the seam's own emit: one 'E27_DFLASH {...draft_load_check...}' line per PP1 worker, 67 = 67",
   L1.startswith("E27_DFLASH ") and '"event": "draft_load_check"' in L1 and json.loads(L1.split(" ", 1)[1])["loaded"] == 67
   and json.loads(L1.split(" ", 1)[1])["expected"] == 67, L1)
TAG = "(Worker_PP1_TP0_EP2 pid=2410) INFO 11-01 08:41:47 [cuda.py:446] Using FLASH_ATTN backend."
for nm_, log_ in (("no prefix (worker print(): the W4 GPU log's NB4_PPQ lines)", f"{TAG}\n{L1}\n{L2}\n"),
                  ("a vLLM worker prefix", f"(Worker_PP1_TP0_EP2 pid=2410) {L1}\n(Worker_PP1_TP1_EP3 pid=2411) {L2}\n"),
                  ("a re-printed line of the same pid", f"{L1}\n{L2}\n{L2}\n")):
    rb_ = N.dflash_readback(ARMS["D3s"], log_)
    ok(f"K7 draft_load_check parses to PASS for 2 PP1 TP workers: {nm_}", rb_["checks"]["draft_load_check"] is True, rb_["rec"].get("draft_load_check"))
for nm_, log_ in (("one worker only", f"{L1}\n"), ("a skipped check (DFLASH_LOAD_CHECK=0)", f"{_emit(2410, True)}\n{_emit(2411, True)}\n"),
                  ("a PP0 worker tag", f"(Worker_PP0_TP0_EP0 pid=2408) {L1}\n(Worker_PP0_TP1_EP1 pid=2409) {L2}\n")):
    rb_ = N.dflash_readback(ARMS["D3s"], log_)
    ok(f"K7 draft_load_check stays a real failure: {nm_}", rb_["checks"]["draft_load_check"] is False, rb_["rec"].get("draft_load_check"))


# ---------------------------------------------------------------- K8 B1: the e2-guard runtime cudagraph check admits the DFlash2 drafter
_GRD = None
for _c in (HERE.parent / "merlin", HERE / "merlin", Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")):
    if (_c / DS.GRD_REL).is_file():
        _GRD = _c / DS.GRD_REL
        break
if _GRD is None:
    ok("K8 Merlin vllm/v1/e2_guards.py found", False)
else:
    import enum
    import hashlib as _hl
    _orig = _GRD.read_text(encoding="utf-8")
    ok("K8 e2_guards.py is Merlin ad42e37's (GRD_PRE)", _hl.sha256(_orig.encode()).hexdigest() == DS.GRD_PRE)
    _patched = DS.apply_unified(_orig, DS.GRD_PATCH)
    ok("K8 the seam's e2_guards edit lands exactly (GRD_POST + marker), and reverses to the original",
       _hl.sha256(_patched.encode()).hexdigest() == DS.GRD_POST and DS.GRD_MARKER in _patched
       and DS.apply_unified(_patched, DS.reverse_unified(DS.GRD_PATCH)) == _orig)

    def _guard(text):
        t = ast.parse(text)
        want = ("EXPECTED_RESOLVED_MODES", "_DRAFT_MANAGER_ATTRS", "_name", "assert_resolved_cudagraph_modes")
        seg = [ast.get_source_segment(text, n) for n in t.body
               if getattr(n, "name", None) in want or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in want)]
        ns_g = {"REQUIRED_CUDAGRAPH_MODE_NAME": "FULL_AND_PIECEWISE", "QWEN4EXP_TARGET_ARCHITECTURES": frozenset({"Qwen4ExpForCausalLM"}),
                "logger": type("L", (), {"info": staticmethod(lambda *a, **k: None)})()}
        exec("\n\n".join(seg), ns_g)
        return ns_g["assert_resolved_cudagraph_modes"]

    class _M(enum.Enum):
        FULL_AND_PIECEWISE = 1
        FULL_DECODE_ONLY = 2
        NONE = 0

    def _mgr(mode):
        return type("Mgr", (), {"cudagraph_mode": mode})()

    class DFlash2Speculator:
        def __init__(self, mode):
            self.query_cudagraph_manager = _mgr(mode)

    _cfg = type("C", (), {"model_config": type("MC", (), {"architectures": ["Qwen4ExpForCausalLM"]})()})()

    def _raises(fn, *a):
        try:
            fn(*a)
            return None
        except RuntimeError as e:
            return str(e)
    g_orig, g_new = _guard(_orig), _guard(_patched)
    r0 = _raises(g_orig, _cfg, _mgr(_M.FULL_AND_PIECEWISE), DFlash2Speculator(_M.FULL_DECODE_ONLY))
    ok("K8 the ORIGINAL guard raises '<unrecognised speculator DFlash2Speculator>' (the GPU failure, reproduced)",
       bool(r0) and "unrecognised speculator DFlash2Speculator" in r0, r0)
    ok("K8 the patched guard: DFlash2 query_cudagraph_manager FULL_DECODE_ONLY -> no raise",
       _raises(g_new, _cfg, _mgr(_M.FULL_AND_PIECEWISE), DFlash2Speculator(_M.FULL_DECODE_ONLY)) is None)
    r1 = _raises(g_new, _cfg, _mgr(_M.FULL_AND_PIECEWISE), DFlash2Speculator(_M.NONE))
    ok("K8 the patched guard: an eager drafter (NONE) still raises", bool(r1) and "draft.query_cudagraph_manager" in r1, r1)
    r2 = _raises(g_new, _cfg, _mgr(_M.FULL_DECODE_ONLY), DFlash2Speculator(_M.FULL_DECODE_ONLY))
    ok("K8 the patched guard: a downgraded target still raises", bool(r2) and "target" in r2, r2)
    _mtp = type("MultiModuleMTPSpeculator", (), {"cudagraph_manager": _mgr(_M.FULL_DECODE_ONLY)})()
    ok("K8 an MTP speculator is judged exactly as before (cudagraph_manager FULL_DECODE_ONLY -> no raise, both versions)",
       _raises(g_new, _cfg, _mgr(_M.FULL_AND_PIECEWISE), _mtp) is None and _raises(g_orig, _cfg, _mgr(_M.FULL_AND_PIECEWISE), _mtp) is None)
    ok("K8 GRD_REL is a core seam file (applied with every DFlash arm)", DS.GRD_REL in DS.CORE_RELS and DS.GRD_REL in DS.PATCHES)

# ---------------------------------------------------------------- K9 B2: only hard breakage stops an arm; heuristics are recorded
_ARC = " ".join(f"Row{r}: col{c}-{c + 3} all 8 col{c + 4} 0 col{c + 5} 8" for r in range(40) for c in range(0, 30, 6))
ok("K9 the GPU case: ARC grid text reads as LOW_VARIETY to the full check", N.degenerate(_ARC) == "LOW_VARIETY", N.degenerate(_ARC))
_toks = list(range(4000))


def _dec(seg):
    return _ARC


rv = N.garbage_review([{"text": _ARC, "tokens": _toks}], _dec, N.degenerate)
ok("K9 a window of ARC grid text -> not garbage (gate CLEAN), LOW_VARIETY recorded as a label", rv["gate"] == "CLEAN"
   and rv["labels"] == ["LOW_VARIETY"] and rv["windowed"] == [], rv)
rv_h = N.garbage_review([{"text": _ARC, "tokens": _toks}], _dec, N.hard_only(N.degenerate))
ok("K9 the same through the notebook's hard-only L.degenerate (its .e27_full records the label)", rv_h["gate"] == "CLEAN"
   and rv_h["labels"] == ["LOW_VARIETY"], rv_h)
w_ = {"status": "FAIL_GARBAGE_UNDER_LOAD", "level": 16, "running_min": 16, "running_max": 16, "running_eq_level": True,
      "prefill_done_before_window": True, "tok_s": 575.0, "preemptions": 0}
N.apply_windowed_gate(w_, rv, 16)
ok("K9 the window keeps STEADY + labels, so its tok/s counts", w_["status"] == "STEADY" and w_["labels"] == ["LOW_VARIETY"]
   and T.window_ok(w_)[0] is True, (w_["status"], T.window_ok(w_)))
w2_ = {"status": "STEADY", "level": 16}
N.apply_windowed_gate(w2_, N.garbage_review([{"text": "", "tokens": [], "error": None}], _dec, N.degenerate), 16)
ok("K9 an EMPTY complete row -> hard garbage: the window fails (FAIL_GARBAGE_UNDER_LOAD)", w2_["status"] == "FAIL_GARBAGE_UNDER_LOAD", w2_)
rep_ = "x" + "7" * 260
ok("K9 REPEATED_CHAR / BANG_GARBAGE stay hard", N.garbage_review([{"text": rep_, "tokens": []}], _dec, N.degenerate)["gate"] == "GARBAGE"
   and N.garbage_review([{"text": "!" * 300, "tokens": []}], _dec, N.degenerate)["gate"] == "GARBAGE")
ok("K9 hard_only: LOW_VARIETY -> None, EMPTY / REPEATED_CHAR -> kept", N.hard_only(N.degenerate)(_ARC) is None
   and N.hard_only(N.degenerate)("") == "EMPTY" and N.hard_only(N.degenerate)(rep_) == "REPEATED_CHAR")
_acc = {"status": "OK", "drafts": 400, "accept_rate": 0.6, "al": 2.8}
_sm = {"errors": [], "garbage": {"gate": "CLEAN"}, "tokens": [2048, 2048], "acceptance": _acc, "ladder": None}
ok("K9 smoke: an EMPTY stream -> FAIL_SMOKE_EMPTY (stop)", N.smoke_gate(dict(_sm, tokens=[0, 2048])) == "FAIL_SMOKE_EMPTY")
ok("K9 smoke: a short stream -> INCONCLUSIVE_SMOKE_SHORT (recorded, not FAIL)", N.smoke_gate(dict(_sm, tokens=[900, 2048])).startswith("INCONCLUSIVE"))
ok("K9 smoke: < 2 % accepted after >= 200 drafts stays FAIL_SMOKE_DFLASH_ZERO_ACCEPT",
   N.smoke_gate(dict(_sm, acceptance=dict(_acc, accept_rate=0.01))) == "FAIL_SMOKE_DFLASH_ZERO_ACCEPT")
ok("K9 smoke: a request error stays FAIL_SMOKE_REQUEST_ERROR", N.smoke_gate(dict(_sm, errors=["boom"])) == "FAIL_SMOKE_REQUEST_ERROR")
ok("K9 long oracle: FAIL_HIGH_MARGIN vs Q -> recorded (never NO_GO / FAIL); a request error stays NO_GO_LONG_ORACLE_ERROR",
   not N.long_oracle_gate({"long_oracle": {"complete": True, "errors": [], "garbage": [], "vs_q_long": {"status": "FAIL_HIGH_MARGIN"}}}, False).startswith(("NO_GO", "FAIL"))
   and N.long_oracle_gate({"long_oracle": {"complete": True, "errors": ["x"], "garbage": []}}, False) == "NO_GO_LONG_ORACLE_ERROR")
_pr_wrong = {"ok": False, "cases": [{"q": "7*6", "ok": False, "content": "41"}]}
_pr_err = {"ok": False, "cases": [{"q": "7*6", "ok": False, "error": "URLError"}]}
_sam_ok = {"tokens": 64, "error": None, "alive_after": True, "garbage": None, "ok": True}
ok("K9 short gate: a wrong arithmetic answer is recorded (no stop); a probe request error stops",
   N.short_gate_hard(_pr_wrong, _sam_ok) is False and N.short_gate_hard(_pr_err, _sam_ok) is True)
ok("K9 short gate: a short sampled request is recorded; no token / an error / a dead server stops",
   N.short_gate_hard({"ok": True}, dict(_sam_ok, tokens=40, ok=False)) is False and N.short_gate_hard({"ok": True}, dict(_sam_ok, tokens=0)) is True
   and N.short_gate_hard({"ok": True}, dict(_sam_ok, error="x")) is True and N.short_gate_hard({"ok": True}, dict(_sam_ok, alive_after=False)) is True)
_o = {"sampled": _sam_ok, "probes": {"ok": True}, "oracle": {"errors": [], "garbage": [None], "complete": True, "vs_q": {"status": "FAIL_HIGH_MARGIN"}}}
ok("K9 quality gate: FAIL_HIGH_MARGIN -> INCONCLUSIVE_HIGH_MARGIN (no NO_GO_QUALITY stop); oracle request error -> FAIL",
   N.quality_gate_hard(_o, "q") == "INCONCLUSIVE_HIGH_MARGIN"
   and N.quality_gate_hard(dict(_o, oracle=dict(_o["oracle"], errors=["x"])), "q") == "FAIL")
ok("K9 quality gate: wrong probe answers -> INCONCLUSIVE_PROBES_WRONG; LOW_VARIETY in the oracle rows -> not FAIL",
   N.quality_gate_hard(dict(_o, probes=_pr_wrong, oracle=dict(_o["oracle"], vs_q={"status": "EXACT"})), "q") == "INCONCLUSIVE_PROBES_WRONG"
   and not N.quality_gate_hard(dict(_o, oracle=dict(_o["oracle"], garbage=["LOW_VARIETY"])), "q").startswith("FAIL"))
ok("K9 correctness: a LOW_VARIETY-labelled STEADY window and a HIGH_MARGIN oracle -> never NO_GO (Q's GPU kill, reproduced and fixed)",
   not str(N.oracle_gate_hard(_o, ("q",), (), [w_]) or "").startswith("NO_GO")
   and N.oracle_gate_hard(_o, ("q",), (), [dict(w_, status="FAIL_GARBAGE_UNDER_LOAD")]) == "NO_GO_GARBAGE_UNDER_LOAD")
if BODY.is_file():
    btxt_ = BODY.read_text(encoding="utf-8")
    fns_ = {n.name: ast.get_source_segment(btxt_, n) for n in ast.parse(btxt_).body if isinstance(n, ast.FunctionDef)}
    ok("K9 the body rebinds the lab's degenerate to hard-only and redefines the four gates",
       "L.degenerate = N.hard_only(L.degenerate)" in btxt_ and all(k in fns_ for k in ("short_gate_failed", "quality_gate", "oracle_gate", "evidence_gate")))
    ns9 = {"N": N, "FAKE": False, "CLEAN": N.CLEAN, "_windows": lambda o: list((o.get("steady") or {}).values())}
    for k in ("short_gate_failed", "quality_gate", "oracle_gate", "evidence_gate"):
        exec(fns_[k], ns9)
    o9 = dict(_o, probes=_pr_wrong, steady={"8k@16#1": w_}, oracle=dict(_o["oracle"], row_count=4, expected_rows=4))
    ok("K9 the body's gates on Q's GPU case (LOW_VARIETY window, HIGH_MARGIN oracle, a wrong probe): no stop, no NO_GO",
       ns9["short_gate_failed"](o9) is False and not ns9["quality_gate"](o9, "q").startswith("FAIL")
       and not str(ns9["oracle_gate"](o9, ("q",)) or "").startswith("NO_GO") and not str(ns9["evidence_gate"](o9, 4) or "").startswith("NO_GO"),
       (ns9["short_gate_failed"](o9), ns9["quality_gate"](o9, "q"), ns9["oracle_gate"](o9, ("q",)), ns9["evidence_gate"](o9, 4)))
    ok("K9 the body's gates: a probe request error / an EMPTY sampled request still stop",
       ns9["short_gate_failed"](dict(o9, probes=_pr_err)) is True and ns9["evidence_gate"](dict(o9, sampled=dict(_sam_ok, tokens=0)), 4) == "NO_GO_SAMPLED_PATH")
print(f"{RES['pass']}/{RES['pass'] + RES['fail']} PASS", flush=True)
sys.exit(1 if RES["fail"] else 0)
