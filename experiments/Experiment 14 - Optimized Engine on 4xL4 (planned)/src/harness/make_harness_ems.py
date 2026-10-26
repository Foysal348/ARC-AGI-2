"""Writes harness_ems_nb.py: the E15P harness setup (fake SGLang server = e15p_fake_server.py, staging, payload dataset
shape, fake sglang package for the E14 patches) + an EMS body that runs all 9 code cells of the EMS notebook and checks
every arm's real launch flags, the gate / windows, the 2 x TP2 servers and the report."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
E15H = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/src/harness")
src = (E15H / "harness_e15p_nb.py").read_text(encoding="utf-8")
head = src[:src.index("nb = json.load(open(NB, encoding=\"utf-8\"))")]
head = head.replace('r"D:/ARC AGI 2/_scratch/e15p/harness_runs"', 'r"D:/ARC AGI 2/_scratch/ems/harness_runs"')
head = head.replace('E11_FAKE_SERVER=str(HERE / "e15p_fake_server.py")', 'E11_FAKE_SERVER=str(E15H / "e15p_fake_server.py")')
head = head.replace('for _f in (Path(NB).resolve().parent / "Build" / "payload").glob("q27x_e14_modules_*.bin"):',
                    'for _f in (Path(os.environ.get("EMS_PAYLOAD_DIR", r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k '
                    'independent attempts (planned)/Experiment Notebooks/Build/payload"))).glob("q27x_e14_modules_*.bin"):')
assert head.count('E15H / "e15p_fake_server.py"') == 1 and head.count("EMS_PAYLOAD_DIR") == 1
head = head.replace('HERE = Path(__file__).resolve().parent',
                    'HERE = Path(__file__).resolve().parent\nE15H = Path(os.environ.get("E15P_HARNESS_DIR", r"' + str(E15H).replace("\\", "/") + '"))', 1)
body = r'''
nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
assert len(code) == 9 and "CELL EMS —" in code[7] and "CELL EMS-R" in code[8], [c.splitlines()[0][:70] for c in code]
G = {"__name__": "__main__"}
cell_sec = []
for i, c in enumerate(code):
    t = time.time()
    exec(compile(c, f"<cell{i}>", "exec"), G)
    cell_sec.append(round(time.time() - t, 1))

checks = []


def check(label, cond, info=""):
    checks.append((label, bool(cond), "" if cond else str(info)[:1200]))


R = json.load(open(WORK / "q27x" / "e14" / "e14_results.json", encoding="utf-8"))
EM = R.get("ems") or {}
A = EM.get("arms") or {}
L = {x["arm"]: x for x in R.get("launches") or [] if "arm" in x}


def fl(arm):
    return [str(x) for x in (L.get(arm) or {}).get("flags") or []]


def val(f, k, n=1):
    return f[f.index(k) + 1:f.index(k) + 1 + n] if k in f else None


check("mode: profile EMS, not a rerun, the matrix ran", G["PROFILE"] == "EMS" and G["EMS_ON"] and not G["RERUN"], G["PROFILE"])
want = ["ref_dflash_w6_s16", "target_only", "mtp_3_1_4", "ntn_fp8", "ntn_int4", "dflash_w6_s20", "dflash_w6_s24",
        "mtp_2_1_3", "mtp_3_1_4_mem86", "mtp_replay_fp16", "mtp_replay_fp32", "dflash_w4", "dflash_w8",
        "tp2x2_dflash_w6"]
check("every planned arm was run and recorded ok", all(k in A and A[k].get("ok") for k in want),
      {k: (A.get(k) or {}).get("ok") for k in want})
check("measured tok/s for every single-server arm (8k and 40k bands)",
      all(A[k].get("short_tok_s") and A[k].get("long_tok_s") for k in want if k in A and not k.startswith("tp2")),
      {k: (A[k].get("short_tok_s"), A[k].get("long_tok_s")) for k in want if k in A})
check("rev 2: target-only is the gate reference; the DFlash reference judged against it (PASS on the fake)",
      EM.get("gate_reference", "").startswith("target_only") and (EM.get("dflash_vs_target_gate") or {}).get("verdict") == "PASS"
      and A.get("ref_dflash_w6_s16", {}).get("gate") == "PASS" and (EM.get("probe_tgt_aa") or {}).get("compared")
      and (EM.get("dflash_vs_target_probe") or {}).get("compared"), (EM.get("gate_reference"), EM.get("dflash_vs_target_gate"),
                                                                     A.get("ref_dflash_w6_s16", {}).get("gate")))
check("rev 2: every gated INT4 arm also compared with the target-only probe",
      all(A[k].get("probe_flips_vs_target") is not None for k in ("mtp_3_1_4", "ntn_int4", "dflash_w4", "dflash_w8") if k in A),
      {k: A.get(k, {}).get("probe_flips_vs_target") for k in ("mtp_3_1_4", "ntn_int4", "dflash_w4", "dflash_w8")})
check("rev 2: the linear-attention arms are dropped with their source reasons, never launched",
      set(EM.get("dropped") or {}) == {"linattn_flashinfer", "linattn_cutedsl"} and not any(k.startswith("linattn") for k in L),
      (EM.get("dropped"), [k for k in L if k.startswith("linattn")]))
check("rev 2: no arm flagged CAPPED on the fake (it admits what it is asked)", not any(A[k].get("capped") for k in A),
      {k: (A[k].get("level_wanted"), A[k].get("per_replica_streams"), A[k].get("level_short")) for k in A if A[k].get("capped")})
check("INT4 arms pass the correctness gate vs the target-only A/A",
      all(A[k].get("gate") == "PASS" for k in ("mtp_3_1_4", "ntn_int4", "dflash_w6_s20", "dflash_w4", "dflash_w8") if k in A),
      {k: A.get(k, {}).get("gate") for k in ("mtp_3_1_4", "ntn_int4", "dflash_w6_s20", "dflash_w4", "dflash_w8")})
check("FP8 is diagnostic, target-only is the reference",
      A.get("ntn_fp8", {}).get("gate") == "diagnostic" and A.get("target_only", {}).get("gate") == "reference",
      (A.get("ntn_fp8", {}).get("gate"), A.get("target_only", {}).get("gate")))
f = fl("mtp_3_1_4")
check("MTP arm: EAGLE 3/1/4, draft path = the INT4 dir, no DFLASH, no delayer flag",
      val(f, "--speculative-algorithm") == ["EAGLE"] and val(f, "--speculative-num-steps") == ["3"]
      and val(f, "--speculative-eagle-topk") == ["1"] and val(f, "--speculative-num-draft-tokens") == ["4"]
      and val(f, "--speculative-draft-model-path") == val(f, "--model-path") and "DFLASH" not in f
      and "--min-free-slots-delay" not in f and "--speculative-draft-model-quantization" not in f, f)
f = fl("ntn_fp8")
check("NTN-FP8: FP8 model + its draft, flashinfer attention, triton linear attention, chunk 2048, max prefill 4096, "
      "extra_buffer, mamba ratio 0.5 + rev 2b mamba 85 (5 x 17), SSM fp32, EAGLE 3/1/4",
      val(f, "--model-path") == [str(G["MODEL_PATH"])] and val(f, "--speculative-draft-model-path") == [str(G["MODEL_PATH"])]
      and val(f, "--attention-backend") == ["flashinfer"] and val(f, "--linear-attn-backend") == ["triton"]
      and val(f, "--chunked-prefill-size") == ["2048"] and val(f, "--max-prefill-tokens") == ["4096"]
      and val(f, "--mamba-radix-cache-strategy") == ["extra_buffer"] and val(f, "--mamba-full-memory-ratio") == ["0.5"]
      and val(f, "--max-mamba-cache-size") == ["85"] and val(f, "--mamba-ssm-dtype") == ["float32"]
      and val(f, "--speculative-algorithm") == ["EAGLE"] and f.count("--model-path") == 1, f)
f = fl("ntn_int4")
check("NTN-INT4: the same flags on the INT4 model", val(f, "--model-path") != [str(G["MODEL_PATH"])]
      and val(f, "--mamba-ssm-dtype") == ["float32"] and val(f, "--chunked-prefill-size") == ["2048"], f)
f = fl("dflash_w6_s20")
check("20 streams: mrr 21, graphs 1..21, mamba 84, DFLASH width 6",
      val(f, "--max-running-requests") == ["21"] and val(f, "--cuda-graph-bs-decode", 21) == [str(b) for b in range(1, 22)]
      and val(f, "--max-mamba-cache-size") == ["84"] and val(f, "--speculative-num-draft-tokens") == ["6"], f)
check("20 / 24 streams really measured at 20 / 24", A.get("dflash_w6_s20", {}).get("level_short") == 20
      and A.get("dflash_w6_s24", {}).get("level_short") == 24, (A.get("dflash_w6_s20", {}).get("level_short"),
                                                                  A.get("dflash_w6_s24", {}).get("level_short")))
check("each flag appears once in every launch (no duplicate after the edits)",
      all(max(fl(k).count(x) for x in fl(k) if x.startswith("--")) == 1 for k in L if fl(k)),
      [k for k in L if fl(k) and max(fl(k).count(x) for x in fl(k) if x.startswith("--")) > 1])
check("width / ReplaySSM arms set what they say",
      val(fl("dflash_w4"), "--speculative-num-draft-tokens") == ["4"] and val(fl("dflash_w8"), "--speculative-num-draft-tokens") == ["8"]
      and val(fl("mtp_replay_fp32"), "--mamba-ssm-dtype") == ["float32"] and "--enable-linear-replayssm-spec" in fl("mtp_replay_fp16"))
_best = [k for k in A if k.startswith("mtp_") and k.endswith("_s20")]
check("the best gated MTP was re-run at 20 streams", len(_best) == 1 and A[_best[0]].get("level_short") == 20, _best)
tp = (R.get("phases") or {}).get("EMS_tp2x2_dflash_w6") or {}
reps = (tp.get("launch") or {}).get("replicas") or []
check("2 x TP2: both replicas ready on their own ports, 8 + 8 streams measured, sum of tok/s recorded",
      A.get("tp2x2_dflash_w6", {}).get("replicas_ready") == [True, True] and len({r["port"] for r in reps}) == 2
      and (A["tp2x2_dflash_w6"].get("short_tok_s") or 0) > 0 and (A["tp2x2_dflash_w6"].get("long_tok_s") or 0) > 0,
      (A.get("tp2x2_dflash_w6"), reps))
_t2 = A.get("tp2x2_dflash_w6", {})
_cd = _t2.get("cache_dirs") or []
check("rev 2: 2 x TP2 pairs from the topology (default 0-1 / 2-3 without nvidia-smi), own cache dirs per replica, "
      "custom all-reduce state recorded", _t2.get("pairs") == [[0, 1], [2, 3]] and len(_cd) == 2
      and all(len({d[k] for d in _cd}) == 2 for k in _cd[0]) and all(v.endswith(("-r0", "-r1")) or "-r0" in v or "-r1" in v
                                                                      for d in _cd for v in d.values())
      and len(_t2.get("custom_allreduce_disabled") or []) == 2 and "pairs" in (EM.get("topology") or {}),
      (_t2.get("pairs"), _cd, _t2.get("custom_allreduce_disabled"), EM.get("topology")))
f = fl("mtp_3_1_4")
check("rev 2b: oracle flag, reference name from the real flags, family tags, MTP gate vs the DFlash reference recorded",
      EM.get("oracle") == "target_only" and EM.get("ref_name") == "ref_dflash_w6_s16" and A["mtp_3_1_4"].get("family") == "mtp"
      and A["ntn_int4"].get("family") == "mtp" and A["dflash_w4"].get("family") == "dflash" and A["target_only"].get("family") is None
      and A["mtp_3_1_4"].get("gate_vs_dflash_ref") == "PASS" and A["mtp_3_1_4"].get("probe_complete_vs_target")
      and A["mtp_3_1_4"].get("server_alive_after") is True and not EM.get("error"),
      (EM.get("oracle"), EM.get("ref_name"), {k: (A[k].get("family"), A[k].get("gate_vs_dflash_ref")) for k in A}, EM.get("error")))
_t2r = (R.get("phases") or {}).get("EMS_tp2x2_dflash_w6") or {}
check("rev 2b: 2 x TP2 = both replicas gated, all four windows ok, KV-for-production flag, custom all-reduce off",
      len(((_t2r.get("result") or {}).get("gate_replicas")) or []) == 2 and (_t2r.get("result") or {}).get("all_windows_ok")
      and "kv_ok_for_production" in A.get("tp2x2_dflash_w6", {})
      and len((_t2r.get("launch") or {}).get("replica_flags") or []) == 2
      and all("--disable-custom-all-reduce" in fl_ and "--cuda-graph-max-bs-decode" not in fl_ for fl_ in (_t2r.get("launch") or {}).get("replica_flags")),
      ((_t2r.get("result") or {}).get("all_windows_ok"), (_t2r.get("launch") or {}).get("replica_flags")))
check("rev 2c: the PR #45151 arm ran with EMS_PR45151=1 (the patch itself is skipped in the harness: no SGLang install)",
      A.get("mtp_3_1_4_pr45151", {}).get("ok") and A["mtp_3_1_4_pr45151"].get("env") == {"EMS_PR45151": "1"}
      and (EM.get("pr45151_patch") or {}).get("why") == "harness: no SGLang install" and ("p2p_read" in (EM.get("topology") or {}) or "error" in (EM.get("topology") or {})),   # no nvidia-smi locally
      (A.get("mtp_3_1_4_pr45151"), EM.get("pr45151_patch"), (EM.get("topology") or {}).keys()))
check("2 x TP2 also ran on the best MTP", any(k.startswith("tp2x2_mtp_") and A[k].get("ok") for k in A), list(A))
check("the 2 x TP2 slot was not skipped (reserved slot)", not any((v or {}).get("skipped") for k, v in (R.get("phases") or {}).items()
                                                                  if k.startswith("EMS_tp2")), R.get("phases", {}).get("EMS_tp2x2_dflash_w6"))
check("report table + bundle written; no server left running",
      (WORK / "q27x" / "e14" / "ems_report.md").exists() and (WORK / "Q27X_EMS_BUNDLE.zip").exists()
      and (G.get("E11_SERVER") is None or not G["E11_SERVER"].alive()))
sub = json.load(open(WORK / "submission.json", encoding="utf-8"))
check("submission.json is the cell-0 fallback for every test task", set(sub) == set(TEST))
print("cell seconds:", cell_sec)
print((WORK / "q27x" / "e14" / "ems_report.md").read_text(encoding="utf-8") if (WORK / "q27x" / "e14" / "ems_report.md").exists() else "no report")
for label, ok, info in checks:
    print(("PASS " if ok else "FAIL ") + label + ("" if ok else "  <-- " + info))
nf = sum(not ok for _, ok, _ in checks)
print(f"\n{nf} failed of {len(checks)}" if nf else f"\nALL {len(checks)} CHECKS PASS (EMS)")
sys.stdout.flush()
os._exit(1 if nf else 0)
'''
(HERE / "harness_ems_nb.py").write_text(head + body, encoding="utf-8", newline="\n")
print("written")
