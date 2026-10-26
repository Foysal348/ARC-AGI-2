"""Writes harness_vms_nb.py: the E15P harness setup (fixtures, payload dataset shape, env) + a VMS body that runs all 8
code cells of the VMS notebook against fake_vllm_server.py and checks every arm's real `vllm serve` flags / env, the
gate, the windows (levels 16 / 20 / 24, 2 x TP2 = 16 over DP2) and the report."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
E15H = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/src/harness")
src = (E15H / "harness_e15p_nb.py").read_text(encoding="utf-8")
head = src[:src.index("nb = json.load(open(NB, encoding=\"utf-8\"))")]
head = head.replace('r"D:/ARC AGI 2/_scratch/e15p/harness_runs"', 'r"D:/ARC AGI 2/_scratch/vms/harness_runs"')
head = head.replace('for _f in (Path(NB).resolve().parent / "Build" / "payload").glob("q27x_e14_modules_*.bin"):',
                    'for _f in (Path(os.environ.get("EMS_PAYLOAD_DIR", r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k '
                    'independent attempts (planned)/Experiment Notebooks/Build/payload"))).glob("q27x_e14_modules_*.bin"):')
body = r'''
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    VPORT = s.getsockname()[1]
os.environ.update(VMS_FAKE_SERVER=str(HERE / "fake_vllm_server.py"), VMS_PORT=str(VPORT), FAKE_SPEED="400")
# rev 2: the RedHat INT4 safetensors index (real key names, kernel_issue/redhat_index.json) for the static MTP check;
# VMS_BAD_INT4=1 = the #58807 case (the ignore list without the mtp entry) -> every INT4 arm must be skipped
_ri = json.load(open(os.environ.get("VMS_REDHAT_INDEX", r"D:/ARC AGI 2/_scratch/kernel_issue/redhat_index.json"), encoding="utf-8"))
json.dump(_ri, open(INT4 / "model.safetensors.index.json", "w"))
BAD_INT4 = os.environ.get("VMS_BAD_INT4") == "1"
if BAD_INT4:
    _c = json.load(open(INT4 / "config.json", encoding="utf-8"))
    _q = _c.get("quantization_config") or _c["text_config"]["quantization_config"]
    _q["ignore"] = [x for x in _q["ignore"] if "mtp" not in x]
    json.dump(_c, open(INT4 / "config.json", "w"))
nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
assert len(code) == 8 and "CELL V1" in code[5] and "CELL V2" in code[6] and "CELL V3" in code[7]
G = {"__name__": "__main__"}
cell_sec = []
for i, c in enumerate(code):
    t = time.time()
    exec(compile(c, f"<cell{i}>", "exec"), G)
    cell_sec.append(round(time.time() - t, 1))
checks = []


def check(label, cond, info=""):
    checks.append((label, bool(cond), "" if cond else str(info)[:1200]))


V = G["VMS"]
A = V["arms"]
L = {x["arm"]: x for x in V["launches"]}


def fl(arm):
    return [str(x) for x in (L.get(arm) or {}).get("flags") or []]


def val(f, k):
    return f[f.index(k) + 1] if k in f else None


want = ["ref_fp8_mtp3_s16", "fp8_target_only_noprefix", "fp8_mtp2", "fp8_mtp4", "int4_mtp3_s16", "fp8_mtp3_s20", "fp8_mtp3_s24",
        "int4_mtp3_s20", "fp8_mtp3_ssm_fp16", "fp8_mtp3_noprefix", "fp8_mtp3_lb_triton", "tp2x2_fp8_mtp3",
        "tp2x2_int4_mtp3"]
if BAD_INT4:
    want = [k for k in want if "int4" not in k]
check("mode: profile VMS, matrix ran, staging off, venv = fake", G["PROFILE"] == "VMS" and G["VMS_ON"]
      and G["E14_STAGING"] is None and V["install"].get("fake"), (G["PROFILE"], V["install"]))
check("rev 2b: no arm recorded an exception (Kaggle CPU kit run1: Thread._stop shadowed -> TypeError on Py3.12)",
      not any(A[k].get("error") for k in A), {k: A[k].get("error") for k in A if A[k].get("error")})
check("every planned arm ran and recorded ok with tok/s in both bands",
      all(k in A and A[k].get("ok") and A[k].get("short_tok_s") and A[k].get("long_tok_s") for k in want),
      {k: (A.get(k, {}).get("ok"), A.get(k, {}).get("short_tok_s"), A.get(k, {}).get("long_tok_s")) for k in want})
check("rev 2: the oracle (FP8 target-only, prefix caching off) is the gate reference; the production reference judged "
      "against it (PASS on the fake); 0 degenerate outputs", V.get("gate_reference", "").startswith("fp8_target_only_noprefix")
      and A["fp8_target_only_noprefix"]["gate"] == "oracle" and (V.get("ref_vs_oracle") or {}).get("verdict") == "PASS"
      and A["ref_fp8_mtp3_s16"]["gate"] == "PASS" and A["fp8_target_only_noprefix"]["aa"]["degenerate"] == 0
      and "--no-enable-prefix-caching" in fl("fp8_target_only_noprefix") and "--speculative-config" not in fl("fp8_target_only_noprefix"),
      (V.get("gate_reference"), V.get("ref_vs_oracle"), A.get("fp8_target_only_noprefix", {}).get("gate")))
check("rev 2: degenerate counts in every gate and window", all(A[k].get("short", {}).get("degenerate") == 0 for k in want if k in A)
      and all((A[k].get("gate_detail") or {}).get("degenerate", 0) == 0 for k in want if k in A),
      {k: (A[k].get("short", {}).get("degenerate"), (A[k].get("gate_detail") or {}).get("degenerate")) for k in want if k in A})
check("rev 2b: --linear-backend triton set; lb_marlin + the old env arm dropped (SM 8.9 auto = Marlin), kernels from the log",
      val(fl("fp8_mtp3_lb_triton"), "--linear-backend") == "triton" and "fp8_mtp3_lb_marlin" not in A and "fp8_mtp3_marlin" not in A
      and not any("VLLM_TEST_FORCE_FP8_MARLIN" in json.dumps(x.get("env") or {}) for x in V["launches"])
      and any("Triton" in s for s in A["fp8_mtp3_lb_triton"].get("kernels") or [])
      and any("Marlin" in s for s in A["ref_fp8_mtp3_s16"].get("kernels") or [])
      and any("lb_marlin" in k for k in V.get("dropped") or {}), (A.get("fp8_mtp3_lb_triton", {}).get("kernels"), V.get("dropped")))
_rows = [json.loads(x) for x in open(G["VMS_DIR"] / "vms_greedy_rows.jsonl", encoding="utf-8")] if (G["VMS_DIR"] / "vms_greedy_rows.jsonl").exists() else []
check("rev 2b: token + margin gate: rows saved with tokens and margins, every gate detail counts large-margin flips",
      _rows and all(r["tokens"] and len(r["margins"]) == len(r["tokens"]) for x in _rows for v in x["rows"].values() for r in v)
      and all("large_margin_flips" in (A[k].get("gate_detail") or {}).get("serial", {}) for k in want if A.get(k, {}).get("gate_detail")),
      (len(_rows), [A[k].get("gate_detail", {}).get("serial") for k in want[:3] if k in A]))
check("rev 2b: one deadline per arm recorded, both bands measured, server alive after, every KV line kept",
      all(A[k].get("deadline_in_sec") and A[k].get("server_alive_after") for k in want if k in A)
      and all(x.get("kv_tokens_all") for x in V["launches"] if x.get("ok")),
      {k: (A[k].get("deadline_in_sec"), A[k].get("server_alive_after")) for k in want if k in A})
_st = V.get("int4_mtp_static") or {}
check("rev 2: INT4 MTP static check " + ("FAILS on the #58807 ignore list and every INT4 arm is skipped" if BAD_INT4 else
      "passes on the RedHat index (15 dense mtp keys, re:^mtp.* in the ignore list)"),
      (not _st.get("ok") and all("static check" in str(A.get(k, {}).get("skipped")) for k in ("int4_mtp3_s16", "int4_mtp3_s20", "tp2x2_int4_mtp3"))
       and not any(fl(k)[:1] == [str(G["SGL_MODEL_PATH"])] for k in L)) if BAD_INT4 else
      (_st.get("ok") and _st.get("mtp_keys") == 15 and _st.get("packed") == 0 and _st.get("mtp_ignore_entries") == ["re:^mtp.*"]
       and not _st.get("not_ignored")), _st)
check("levels really reached: 16 / 20 / 24, 2 x TP2 = 16 (8 per DP rank)",
      A["ref_fp8_mtp3_s16"]["level_short"] == 16 and A["fp8_mtp3_s20"]["level_short"] == 20
      and A["fp8_mtp3_s24"]["level_short"] == 24 and A["tp2x2_fp8_mtp3"]["level_short"] == 16,
      {k: A[k].get("level_short") for k in want if k in A})
f = fl("ref_fp8_mtp3_s16")
check("reference = S1-60K engine: FP8 model, TP4, fp8 KV, MTP 3, FLASHINFER, prefix caching, 16 seqs, 4096 batched, "
      "mem 0.90, custom all-reduce off", f[0] == str(G["MODEL_PATH"]) and val(f, "--tensor-parallel-size") == "4"
      and val(f, "--kv-cache-dtype") == "fp8_e4m3" and json.loads(val(f, "--speculative-config")) == {"method": "mtp", "num_speculative_tokens": 3}
      and val(f, "--attention-backend") == "FLASHINFER" and "--enable-prefix-caching" in f and val(f, "--max-num-seqs") == "16"
      and val(f, "--max-num-batched-tokens") == "4096" and val(f, "--gpu-memory-utilization") == "0.9"
      and "--disable-custom-all-reduce" in f, f)
check("arms set what they say", json.loads(val(fl("fp8_mtp4"), "--speculative-config"))["num_speculative_tokens"] == 4
      and (BAD_INT4 or fl("int4_mtp3_s16")[0] == str(G["SGL_MODEL_PATH"])) and val(fl("fp8_mtp3_s24"), "--max-num-seqs") == "24"
      and val(fl("fp8_mtp3_ssm_fp16"), "--mamba-ssm-cache-dtype") == "float16" and "--no-enable-prefix-caching" in fl("fp8_mtp3_noprefix")
      and val(fl("tp2x2_fp8_mtp3"), "--tensor-parallel-size") == "2" and val(fl("tp2x2_fp8_mtp3"), "--data-parallel-size") == "2"
      and val(fl("tp2x2_fp8_mtp3"), "--max-num-seqs") == "8" and val(fl("tp2x2_fp8_mtp3"), "--api-server-count") == "1"
      and "--disable-custom-all-reduce" in fl("tp2x2_fp8_mtp3") and "--disable-custom-all-reduce" in fl("ref_fp8_mtp3_s16"))
check("each flag once per launch", all(max(fl(k).count(x) for x in fl(k) if x.startswith("--")) == 1 for k in L if fl(k)))
check("gates: FP8 arms PASS vs the oracle; INT4 diagnostic",
      all(A[k]["gate"] == "PASS" for k in ("fp8_mtp2", "fp8_mtp4", "fp8_mtp3_s20", "fp8_mtp3_lb_triton", "tp2x2_fp8_mtp3"))
      and (BAD_INT4 or A["int4_mtp3_s16"]["gate"] == "diagnostic"),
      {k: A[k].get("gate") for k in want if k in A})
check("accept length measured for MTP arms, none for target-only", A["ref_fp8_mtp3_s16"]["accept_short"]
      and A["fp8_target_only_noprefix"]["accept_short"] is None,
      (A["ref_fp8_mtp3_s16"]["accept_short"], A["fp8_target_only_noprefix"]["accept_short"]))
check("KV tokens parsed from the server log", all(A[k].get("kv_tokens") for k in want if k in A))
check("2 x TP2 not skipped (reserved slot)", not A["tp2x2_fp8_mtp3"].get("skipped") and (BAD_INT4 or not A["tp2x2_int4_mtp3"].get("skipped")))
check("report + bundle; server stopped", (G["VMS_DIR"] / "vms_report.md").exists() and (WORK / "Q27X_VMS_BUNDLE.zip").exists()
      and (G["VMS_SERVER"] is None or not G["VMS_SERVER"].alive()))
sub = json.load(open(WORK / "submission.json", encoding="utf-8"))
check("submission.json = the cell-0 fallback for every test task", set(sub) == set(TEST))
print("cell seconds:", cell_sec)
print((G["VMS_DIR"] / "vms_report.md").read_text(encoding="utf-8") if (G["VMS_DIR"] / "vms_report.md").exists() else "no report")
for label, ok, info in checks:
    print(("PASS " if ok else "FAIL ") + label + ("" if ok else "  <-- " + info))
nf = sum(not ok for _, ok, _ in checks)
print(f"\n{nf} failed of {len(checks)}" if nf else f"\nALL {len(checks)} CHECKS PASS (VMS)")
sys.stdout.flush()
os._exit(1 if nf else 0)
'''
(HERE / "harness_vms_nb.py").write_text(head + body, encoding="utf-8", newline="\n")
print("written")
