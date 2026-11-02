# E28 W4 CPU kit: the code build_e28_w4_cpu_kit.py injects into the NB6Q kit builder (a TEMPLATE: __E28UNITFILES__, __E2EE28__,
# __REALE28__, __NB6QSHA__ are filled in by the kit builder; executed in the NB6Q kit builder's namespace right before its
# `NB6_RAW = NB6_NB.read_bytes()` line, i.e. after NB6Q's own edits of the NB6 kit text S and of the fake server FAKE).
# ================================================================ E28 kit (injected): edits of the NB6 kit text after NB6Q's
E28_UNIT_FILES = __E28UNITFILES__
_E27P = __E27PLATFORMREPR__
# ---- K2 units: a self-contained E28 units body (e28w4_unit_tests.py T1-T7); the NB5 QUALITY regression of the NB6 / NB6Q units
# kits is not carried (NB5 QUALITY 9c71ba89 is unchanged and was ALL OK in both; it + this notebook exceed one 400-kB kit)
S = once(S, """UNIT_FILES = {"nb6.ipynb": NB_TXT, "nb5q.ipynb": NB5Q_TXT, "nb6_unit_tests.py": K5["src"]("nb6_unit_tests.py"), **PUSH_FILES,
              **NB6Q_UNIT_FILES,   # NB6Q kit K2
              **{k: v for k, v in UF.items() if k not in ("nb5.ipynb", "nb5_e2e.py", "nb5_fake_server.py", "make_fake.py", "parts.json")}}""",
         """UNIT_FILES = {"nb6.ipynb": NB_TXT, **PUSH_FILES, **NB6Q_UNIT_FILES}   # E28 kit K2: the notebook, the push folders, the E28 unit files""",
         "E28 K2 unit files")
S = once(S, "KITS = {\n", "UNITS = " + repr(__E28UNITSBODY__) + "   # E28 kit K2: the E28 units body\nKITS = {\n", "E28 K2 units body")
# ---- K3 e2e driver
S = once(S, 'BUDGET = int(os.environ.get("NB6_E2E_BUDGET_RETRY", 3000)) if RETRY else int(os.environ.get("NB6_E2E_BUDGET", 1800))',
         'BUDGET = int(os.environ.get("NB6_E2E_BUDGET_RETRY", 3000)) if RETRY else int(os.environ.get("NB6_E2E_BUDGET_COMMIT" if MODE == "commit" else "NB6_E2E_BUDGET", 1800))   # E28 kit K3',
         "E28 K3 budget")
S = once(S, 'for k in [k for k in os.environ if k.startswith(("NB5", "E12_", "E15B_", "FAKE_", "Q27X_", "NB6Q_", "NB4_"))] + [   # NB6Q kit K3',
         'for k in [k for k in os.environ if k.startswith(("NB5", "E12_", "E15B_", "FAKE_", "Q27X_", "NB6Q_", "NB4_", "E28_"))] + [   # E28 kit K3',
         "E28 K3 env pop")
S = once(S, 'os.environ["NB6_HB_SEC"] = "5"                                                        # the heartbeat loop runs in the e2e\n',
         'os.environ["NB6_HB_SEC"] = "5"                                                        # the heartbeat loop runs in the e2e\n'
         'os.environ["E28_B2_ATTEMPT"] = "9000"                                                 # E28 kit K3: the B2 budget (harness only)\n'
         'os.environ["E28_I8_GATE_IN_FAKE"] = "1"                                               # E28 kit v3: the I8 gate runs on the fake\n'
         'if RETRY:\n'
         '    os.environ["FAKE_I8_NO_LINES"] = "1"            # E28 kit v3: no INT8 build line -> SPEC_ONLY -> the same pin, BF16 head\n',
         "E28 K3 b2")
S = once(S, '''if RR:
    os.environ.update(KAGGLE_IS_COMPETITION_RERUN="1", Q27X_TIME_BUDGET=str(BUDGET))     # the fake deadline
''', '''if RR:
    os.environ.update(KAGGLE_IS_COMPETITION_RERUN="1", Q27X_TIME_BUDGET=str(BUDGET))     # the fake deadline
else:
    os.environ["Q27X_TIME_BUDGET"] = str(BUDGET)       # E28 kit K3: E28EVAL120's baked 412 min -> a fake short deadline
''', "E28 K3 commit budget")
S = once(S, 'os.environ["FAKE_OOM_ABOVE_BYTES"] = str(2479838700 + int(0.8 * 2 ** 30))         # RC2 FP8 pin + 0.8 GiB',
         'os.environ["FAKE_OOM_ABOVE_BYTES"] = str(2479838700 + int(0.6 * 2 ** 30))         # E28 kit K3: RC2 FP8 pin + 0.6 GiB (0.75 fails, 0.5 boots)',
         "E28 K3 oom")
S = once(S, 'want_profile = "RERUN_PROD" if RR else "NB6COMMIT"', 'want_profile = "RERUN_PROD" if RR else "E28EVAL120"   # E28 kit K3', "E28 K3 profile")
S = once(S, '''          lambda: len(boots) == 1 and [t_["pin_gib"] for t_ in b0["tries"]] == [1.0, 0.75] and b0["tries"][0]["ladder"]["failed"]
          and b0["tries"][0]["ladder"]["oom_lines"] and b0["tries"][1].get("status") in (None, "DONE") and b0["pin_used"]["gib"] == 0.75''',
         '''          lambda: len(boots) == 1 and [t_["pin_gib"] for t_ in b0["tries"]] == [0.75, 0.5, 0.5] and b0["tries"][0]["ladder"]["failed"]   # E28 kit K3 / v3
          and b0["tries"][0]["ladder"]["oom_lines"] and b0["tries"][1].get("status") == "I8_READBACK_UNVERIFIED"
          and b0["tries"][-1].get("status") in (None, "DONE") and b0["pin_used"]["gib"] == 0.5''',
         "E28 K3 retry pins")
S = once(S, '"before any phase, attempt 2 (pin +0.75 GiB, the same lever env) booted', '"before any phase, attempt 2 (E28: pin +0.5 GiB after the I8 pin +0.75, the same lever env) booted', "E28 K3 retry label")
S = once(S, '_cw = (0.75, 197, 5, 8, 8) if RETRY else (1.0, 214, 5, 9, 9)     # pin, fake pool, F, launch seqs, workers',
         '_cw = (0.5, 181, 5, 8, 8) if RETRY else (0.75, 197, 5, 12, 12)  # E28 kit K3 / v4: pin, fake pool, F, launch seqs, workers (C12 on 0.75 only)',
         "E28 K3 capacity")
S = once(S, 'f"rule), launch max_num_seqs {_cw[3]} (seqs table), workers = min({_cw[1]} // (16 + {_cw[2]}), {_cw[3]}) = {_cw[4]}",',
         'f"rule), launch max_num_seqs {_cw[3]} (E28: C12 on the I8 pin 0.75, NB6Q\'s table below), workers = {_cw[4]} (E28: 12 on the C12 rung, "\n      f"NB6Q\'s slots rule min({_cw[1]} // (16 + F), seqs) below)",', "E28 K3 capacity label")
# kit v4 (kits v3, notebook af80f094: e2e_rerun-retry NOT OK): the NB6 kit asserts exactly ONE ready launch with the lever env; in
# E28's rerun-retry the 0.5 I8 try ALSO reaches ready (the I8 gate then stops it: I8_READBACK_UNVERIFIED) before the same-pin BF16
# retry boots -> two ready launches, both with the lever env (KIT defect: the expectation predates the I8 gate)
S = once(S, '''      lambda: [L_["env_levers"] for L_ in launches if L_["outcome"] == "ready"] == [_lv] and b0["server_env_levers"] == _lv,''',
         '''      lambda: [L_["env_levers"] for L_ in launches if L_["outcome"] == "ready"] == [_lv] * (2 if RETRY else 1)   # E28 kit v4: the I8 try
      and b0["server_env_levers"] == _lv,                     # the gate stopped + its same-pin BF16 retry (rerun-retry)''', "E28 v4 lever ready launches")
S = once(S, 'SOLVE_IDS = sorted(test) if RR else sorted(P.get("task_list") or [])',
         'SOLVE_IDS = sorted(test) if RR else sorted(G.get("TASK_IDS") or [])   # E28 kit K3: E28EVAL120 = all 120 eval tasks', "E28 K3 solve ids")
S = once(S, """    check("commit: the eval subset (NB6COMMIT task_list) scored by exact match, >= 1 test query hit (the fake's lookup)",
          lambda: P["source"] == "eval" and sorted(G["TASK_IDS"]) == sorted(P["task_list"]) and sc["eval_query_total"] >= len(P["task_list"])
          and sc["eval_query_hits"] >= 1 and set(main["task_solve"]) == set(P["task_list"]),""",
         """    check("commit: E28EVAL120 = all 120 eval tasks (n_tasks 120, no task_list) scored by exact match, >= 1 test query hit",
          lambda: P["source"] == "eval" and P["n_tasks"] == 120 and "task_list" not in P and len(G["TASK_IDS"]) == 120
          and sc["eval_query_total"] >= 120 and sc["eval_query_hits"] >= 1 and set(main["task_solve"]) == set(G["TASK_IDS"]),""",
         "E28 K3 commit score")
S = once(S, """    check("commit: the baked 50-min budget (NB6COMMIT, no override; NB6Q kit K4: the NB6 kit's 40-min expectation was stale)",
          lambda: G["TOTAL_BUDGET_SEC"] == 3000, G.get("TOTAL_BUDGET_SEC"))""",
         """    check(f"commit: the fake short deadline in force (TOTAL_BUDGET_SEC == {BUDGET}); the baked E28EVAL120 budget 412 min, HARD_STOP margin "
          "2 min, the fixed 6-h window (shortened to the room: never skipped)", lambda: G["TOTAL_BUDGET_SEC"] == BUDGET
          and P["time_budget_sec"] == 412 * 60 and P["hard_margin_sec"] == 120 and P["fixed_window_sec"] == 21600
          and (main.get("window_plan") or {}).get("unequal") is True, (G.get("TOTAL_BUDGET_SEC"), main.get("window_plan")))""",
         "E28 K3 commit budget check")
S = once(S, """bad = [c for c in checks if not c[1]]\nfor label, ok, info in checks:""", __E2EE28__ + """bad = [c for c in checks if not c[1]]\nfor label, ok, info in checks:""",
         "E28 K4 e2e checks")
S = once(S, '''"NB6_E2E_ROOT": str(K / "e2e"), "NB6Q_TREE_SRC": str(MTQ), "NB6_E2E_BUDGET": "1800",''',
         '''"NB6_E2E_ROOT": str(K / "e2e"), "NB6Q_TREE_SRC": str(MTQ), "NB6_E2E_BUDGET": "1800", "NB6_E2E_BUDGET_COMMIT": "2700",''', "E28 K3 kit env")
# ---- K3 the kit-local fake server (FAKE ONLY: the E28 launch knobs, no engine claim)
FAKE = once(FAKE, '''if MAXLEN != 131072:
    print(f"ValueError: fake Merlin: --max-model-len {MAXLEN} (NB5 needs 131072)", flush=True)''', '''if MAXLEN not in (131072, 262144):   # E28 kit K3: the E28 server runs 262144
    print(f"ValueError: fake Merlin: --max-model-len {MAXLEN} (NB5 needs 131072, E28 262144)", flush=True)''', "E28 fake maxlen")
FAKE = once(FAKE, '''log(f"[api_utils.py:286] non-default args: {_nd!r}")\n''', '''if "--language-model-only" in argv:          # E28 kit K3 (FAKE ONLY): the text-only flag in the non-default args
    _nd["language_model_only"] = True
if os.environ.get("DRAFT_HEAD") == "int8" and isinstance(SPEC, dict):   # E28 kit K3 (FAKE ONLY): Merlin's I8 knob -> the spec
    SPEC["draft_lm_head_quant"] = "int8"
    _nd["speculative_config"] = SPEC
log(f"[api_utils.py:286] non-default args: {_nd!r}")
if os.environ.get("DRAFT_HEAD") == "int8" and os.environ.get("FAKE_I8_NO_LINES") != "1":    # E28 kit K3 (FAKE ONLY): NB4T's PP1 lines
    for _i8w in (0, 1):
        print(f"(Worker_PP1_TP{_i8w}_EP{_i8w} pid={os.getpid() * 10 + _i8w}) INFO 11-01 00:00:00 [draft_lm_head_quant.py:118] Draft lm_head: "
              "INT8 weight-only copy of the 124160x2560 shard (307.9 MiB; the target keeps its torch.bfloat16 head)", flush=True)
''', "E28 fake i8")
FAKE = once(FAKE, '''"env_nb4": {k: os.environ.get(k) for k in ("NB4_PPDEFER", "NB4_PPACK")}, **kw}) + "\\n")   # NB6Q kit K3''',
            '''"env_nb4": {k: os.environ.get(k) for k in ("NB4_PPDEFER", "NB4_PPACK")},
                                "env_e28": {k: os.environ.get(k) for k in ("DRAFT_HEAD", "E28_SCHED")}, **kw}) + "\\n")   # NB6Q / E28 kit K3''',
            "E28 fake launch env")
compile(FAKE, "nb5_fake_server.py (E28 kit)", "exec")
# ---- K5 real
S = once(S, '''        tries = L5.pin_tries(g["NB5_KV_BYTES"][kv], None) if P_.get("pin_ladder") else [{"gib": 0.0, "kv_bytes": g["NB5_KV_BYTES"][kv]}]''',
         '''        tries = L5.pin_tries(g["NB5_KV_BYTES"][kv], P_.get("pin_start_gib")) if P_.get("pin_ladder") else [{"gib": 0.0, "kv_bytes": g["NB5_KV_BYTES"][kv]}]   # E28 kit K5: the I8 pin first''',
         "E28 K5 pins")
S = once(S, '''            seqs = L5.seqs_for(kv, tr["gib"]) if g["NB5_SEQS_RULE"] == "table" else None''',
         '''            seqs = (int(P_["seqs_fixed"]) if P_.get("seqs_fixed") and int(tr["kv_bytes"]) == int(P_.get("seqs_fixed_kv_bytes") or -1)
                    else (L5.seqs_for(kv, tr["gib"]) if g["NB5_SEQS_RULE"] == "table" else None))   # E28 kit K5 / v4: C12 on the I8 pin only''',
         "E28 K5 seqs")
S = once(S, '''                if str(al.get("--max-model-len")) != "131072": chk.append(f"max-model-len {al.get('--max-model-len')}")''',
         '''                if str(al.get("--max-model-len")) != "131072": chk.append(f"max-model-len {al.get('--max-model-len')}")   # E28 kit K5 (NB6Q's)
                if "--language-model-only" not in [str(x) for x in s.argv]: chk.append("no --language-model-only")
                try:
                    _e28sp = json.loads(spec) if spec not in (None, "None") else {}
                except Exception:
                    _e28sp = {"_unparsable": str(spec)[:80]}
                if _e28sp.get("draft_lm_head_quant") != "int8": chk.append(f"I8: spec draft_lm_head_quant {_e28sp.get('draft_lm_head_quant')!r}")
                if ((g["R"]["setup"].get("merlin_plans") or {}).get(name) or {}).get("build_env") != {"DRAFT_HEAD": "int8"}: chk.append("I8: build env not recorded")''',
         "E28 K5 argv")
S = once(S, '''        if x.get("max_model_len") != 131072: bad.append(f"max_model_len {x.get('max_model_len')}")''',
         '''        if x.get("max_model_len") != 131072: bad.append(f"max_model_len {x.get('max_model_len')}")   # E28 kit K5 (NB6Q's)
        if x.get("language_model_only") is not True: bad.append(f"language_model_only {x.get('language_model_only')}")
        if not (isinstance(x.get("speculative_config"), dict) and x["speculative_config"].get("draft_lm_head_quant") == "int8"):
            bad.append(f"I8 spec {str(x.get('speculative_config'))[:120]}")''', "E28 K5 argparse")
S = once(S, '''    "profile_commit_NB6COMMIT": G.get("PROFILE") == "NB6COMMIT" and G.get("RERUN") is False,''',
         '''    "profile_commit_E28EVAL120": G.get("PROFILE") == "E28EVAL120" and G.get("RERUN") is False,   # E28 kit K5''', "E28 K5 profile")
S = once(S, '''NB5L = G.get("NB5L")             # (the NB5 kit's commit-turn section below uses G / NB5L)\n''',
         __REALE28__ + '''NB5L = G.get("NB5L")             # (the NB5 kit's commit-turn section below uses G / NB5L)\n''', "E28 K5 real section")
S = once(S, '''    "rerun_cells_read_test_file_only_at_COMPD_never_eval_solutions": bool(RERUN_OPENS["test_challenges"])''',
         '''    "e28_sched_applied_both_namespaces_real_payload": (E28R.get("sched") or {}).get("commit") == "APPLIED" and E28R["sched"].get("rerun") == "APPLIED",
    "e28_sched_verify_revert_reapply_real_tree": (E28R.get("sched") or {}).get("roundtrip") is True,
    "e28_imported_arc2_duck_has_the_e28_handover": (E28R.get("sched") or {}).get("handover") is True,
    "e28_specconfig_accepts_draft_lm_head_quant_int8": (E28R.get("spec") or {}).get("ok") is True,
    "rerun_cells_read_test_file_only_at_COMPD_never_eval_solutions": bool(RERUN_OPENS["test_challenges"])''', "E28 K5 verdict")
# ================================================================ E28 kit v2 (kits v1 on Kaggle: four KIT defects, the notebook unchanged)
# V2-1 (e2e capacity): the fake prints the max-concurrency factor with 2 decimals (as vLLM); at --max-model-len 262144 the factor is
# small (197 / 69 = 2.86), so F_log = pool / X - 64 reads 4.88 (pin 0.75) / 5.08 (pin 0.5): the measured rule then gives F 5 / 6.
# The check now asserts F_log within 0.15 of the fake's F 5 and F_eff = NB5's own rule ceil(F_log - 0.05).
S = once(S, '''      and abs(cap["f_log"] - _cw[2]) < 0.06 and cap["f_eff"] == _cw[2] and cap["caps"]["max_num_seqs"] == _cw[3] and cap["workers"] == _cw[4],''',
         '''      and abs(cap["f_log"] - _cw[2]) < 0.15 and cap["f_eff"] == max(1, int(__import__("math").ceil(cap["f_log"] - 0.05)))   # E28 kit v2 V2-1
      and cap["caps"]["max_num_seqs"] == _cw[3] and cap["workers"] == _cw[4],''', "E28 V2-1 capacity")
# V2-3 (real): rerun vs commit plans differ only by the two --port values (Merlin's own + the notebook's): strip every occurrence
S = once(S, '''    for f in ("--port",):
        if f in a:
            i = a.index(f); a[i + 1] = "*"
    return a''', '''    for i in [j for j, x in enumerate(a[:-1]) if x == "--port"]:   # E28 kit v2 V2-3: EVERY --port value (Merlin's + the notebook's)
        a[i + 1] = "*"
    return a''', "E28 V2-3 strip ports")
# V2-2 (real): the argparse probe was written as K/argparse.py: it shadowed the stdlib argparse for every later probe in K (torch's
# own `import argparse` re-ran it: the circular-import / GradBucket / IndexError failures). Never a stdlib name; the fake-L4
# platform (E27 NBDF kit) as the fallback when the plain CPU import fails.
S = once(S, '''    r = venv_py(ARGPARSE, [str(K / "argvs.json")], name="argparse", extra_env=CPU_ENV)''',
         '''    for _old in ("argparse.py", "argparse.pyc"):                     # E28 kit v2 V2-2: never a stdlib module name in K
        (K / _old).unlink(missing_ok=True)
    r = venv_py(ARGPARSE, [str(K / "argvs.json")], name="vllm_argparse_probe", extra_env=CPU_ENV)
    if not r["json"] or "import_error" in r["json"]:
        R6["cpu_first"] = r["json"] or {"tail": r["tail"]}
        r = venv_py(@@E27P@@ + ARGPARSE, [str(K / "argvs.json")], name="vllm_argparse_probe_l4", extra_env=e28_l4_env())'''.replace("@@E27P@@", repr(_E27P)),
         "E28 V2-2 argparse")
S = once(S, '''R6 = {}
try:
    argvs = ''', '''R6 = {}
def e28_l4_env():
    """E28 kit v2: the E27 NBDF kit's env for a fake-L4 probe: no visible GPU + the CUDA toolkit's libcuda stub (when present)."""
    env = {"CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
    try:
        import glob
        c = []
        for root in [(G.get("MERLIN_ENG") or {}).get("cuda_home") or "", "/usr/local/cuda"]:
            if root:
                c += glob.glob(f"{root}/lib*/stubs/libcuda.so") + glob.glob(f"{root}/**/stubs/libcuda.so", recursive=True)
        if c:
            (K / "cudastub").mkdir(exist_ok=True)
            if not (K / "cudastub" / "libcuda.so.1").exists():
                os.symlink(c[0], K / "cudastub" / "libcuda.so.1")
            env["LD_LIBRARY_PATH"] = str(K / "cudastub") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    except Exception:
        pass
    return env
try:
    argvs = ''', "E28 V2-2 l4 env")
# V2-4 (real): the commit profile is E28EVAL120 (the NB6 kit expected NB6COMMIT for the fallback-submission check)
S = once(S, '''        and sc.get("profile") == "NB6COMMIT",''', '''        and sc.get("profile") == "E28EVAL120",   # E28 kit v2 V2-4''', "E28 V2-4 submission profile")
