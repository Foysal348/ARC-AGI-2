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
        "int4_mtp3_s20", "fp8_mtp3_ssm_fp16", "fp8_mtp3_noprefix", "fp8_mtp3_lb_triton", "fp8_mtp3_pcie", "tp2x2_fp8_mtp3",
        "tp2x2_int4_mtp3"]
if BAD_INT4:
    want = [k for k in want if "int4" not in k]
check("mode: profile VMS, matrix ran, staging off, venv = fake", G["PROFILE"] == "VMS" and G["VMS_ON"]
      and G["E14_STAGING"] is None and V["install"].get("fake"), (G["PROFILE"], V["install"]))
check("rev 2b: no arm recorded an exception (Kaggle CPU kit run1: Thread._stop shadowed -> TypeError on Py3.12)",
      not any(A[k].get("error") for k in A), {k: A[k].get("error") for k in A if A[k].get("error")})
check("rev 2d: XL band (~85k prompts) measured on every TP4 arm at the streams the KV holds, none on DP2 x TP2; budget 260 min",
      all(A[k].get("xl_tok_s") and A[k].get("level_xl") for k in want if k in A and not k.startswith("tp2x2"))
      and not any(A[k].get("xl_tok_s") for k in A if k.startswith("tp2x2")) and V["prompts"].get("xl_p50", 0) > 60000
      and G["P"]["time_budget_sec"] == 260 * 60,
      ({k: (A[k].get("xl_tok_s"), A[k].get("level_xl")) for k in A}, V.get("prompts"), G["P"].get("time_budget_sec")))
check("rev 2d: the PCIe-IPC arm set VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 and the log shows it initialized",
      (L.get("fp8_mtp3_pcie") or {}).get("env") == {"VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "1"}
      and (A.get("fp8_mtp3_pcie", {}).get("pcie_ipc") or {}).get("initialized") and not A["ref_fp8_mtp3_s16"].get("pcie_ipc"),
      (L.get("fp8_mtp3_pcie"), A.get("fp8_mtp3_pcie", {}).get("pcie_ipc")))
check("rev 2e: the oracle is ready; FP8 MTP arms PASS_EXACT, XL gate PASS, 0 canary leaks, adoptable; INT4 arms never adoptable",
      V.get("oracle_ready") and all(A[k].get("gate_class") == "PASS_EXACT" and A[k].get("xl_gate") == "PASS"
                                    and (A[k].get("canary") or {}).get("foreign_leaks") == 0 and A[k].get("adoptable")
                                    for k in ("fp8_mtp2", "fp8_mtp4", "fp8_mtp3_s20", "fp8_mtp3_noprefix", "fp8_mtp3_lb_triton"))
      and A["ref_fp8_mtp3_s16"].get("adoptable") and not any(A[k].get("adoptable") for k in A if k.startswith("int4") or k.startswith("tp2x2_int4")),
      {k: (A[k].get("gate_class"), A[k].get("xl_gate"), (A[k].get("canary") or {}).get("foreign_leaks"), A[k].get("adoptable")) for k in A})
check("rev 2e: every window clean: prefill settled, idle after, threads joined, 0 stream errors / empties; e2e tok/s and "
      "prefix-cache counters recorded (0 hits on the no-prefix arm)",
      all((A[k].get(b) or {}).get("prefill_settled") and (A[k].get(b) or {}).get("idle_after") and (A[k].get(b) or {}).get("threads_not_joined") == 0
          and (A[k].get(b) or {}).get("stream_errors") == 0 and (A[k].get(b) or {}).get("empty_outputs") == 0
          and (A[k].get(b) or {}).get("end_to_end_tok_s") for k in want if k in A for b in ("short", "long"))
      and (A["ref_fp8_mtp3_s16"].get("long") or {}).get("prefix_cache_hits", 0) > 0 and (A["fp8_mtp3_noprefix"].get("long") or {}).get("prefix_cache_hits") == 0,
      {k: {b: {x: (A[k].get(b) or {}).get(x) for x in ("prefill_settled", "idle_after", "stream_errors", "empty_outputs", "end_to_end_tok_s", "prefix_cache_hits")}
           for b in ("short", "long")} for k in want[:3] if k in A})
# rev 2e: the audits' negative tests on the notebook's own functions (synthetic rows, no server)
_dv, _jd, _dg, _tb = G["vms_diverge"], G["vms_judge"], G["vms_degenerate"], G["vms_tail_bad"]
_row = lambda ids, done=True, fin="length", m=None, text="x": {"tokens": ids, "margins": m or [2.0] * len(ids), "error": None,
                                                              "done_seen": done, "finish_reason": fin, "text": text}
_inc = _dv([_row([1, 2, 3])], [_row([1, 2, 3], done=False, fin=None)])
_ids_same_text = _dv([_row([10, 11])], [_row([10, 12])])
# rev 2g: the planned row counts (6 serial, every prompt all at once; GPT R5): fewer rows are INCOMPLETE by design
_NS, _NC = min(6, len(G["_S"])), len(G["_S"])
_base = {"serial_a": [_row(list(range(40)))] * _NS, "serial_b": [_row(list(range(40)))] * _NS,
         "conc_a": [_row(list(range(40)))] * _NC, "conc_b": [_row(list(range(40)))] * _NC}
_enc = lambda s: G["tokenizer"].encode(s, add_special_tokens=False)
_bad = [_row([0, 1] + _enc("!" * 2000), m=[2.0, 0.2] + [2.0] * len(_enc("!" * 2000)))]
_grid = _enc(chr(10).join(" ".join("0" for _ in range(30)) for _ in range(12)))
_v_bad, _d_bad = _jd(_bad + _base["serial_a"][1:], _bad + _base["conc_a"][1:], _base)
_v_ex, _d_ex = _jd(_base["serial_a"], _base["conc_a"], _base)
check("rev 2e negative tests: incomplete SSE row = error; different ids = divergence; a near-tie divergence followed by a "
      "degenerate tail FAILS; identical rows = PASS_EXACT; 1 empty + 1 '!'x100 = 2 degenerate",
      _inc["errors"] == 1 and _inc["identical"] == 0 and _ids_same_text["differ"] == 1 and _v_bad == "FAIL" and _tb(_enc("!" * 2000)) and not _tb(_grid)
      and _v_ex == "PASS" and _d_ex.get("gate_class") == "PASS_EXACT" and _dg(["", "!" * 100, "normal reasoning text " * 20, chr(10).join(" ".join("0" for _ in range(30)) for _ in range(12))]) == 2,
      (_inc, _ids_same_text, _v_bad, _d_bad.get("gate_class"), _v_ex, _d_ex.get("gate_class")))
check("rev 2e-d: certified_exact = adoptable and PASS_EXACT on every arm (FP8 MTP arms certified, INT4 never)",
      all(bool(A[k].get("certified_exact")) == bool(A[k].get("adoptable") and A[k].get("gate_class") == "PASS_EXACT") for k in A)
      and A["fp8_mtp2"].get("certified_exact") and not any(A[k].get("certified_exact") for k in A if k.startswith("int4")),
      {k: A[k].get("certified_exact") for k in A})
# rev 2g: the GPT rev 2f audit negative tests, on the notebook's own functions (synthetic copies of a real arm)
import copy as _cp, tempfile as _tf
_cl, _wh, _txb, _ist = G["vms_classify"], G["vms_window_healthy"], G["vms_text_bad"], G["vms_int4_mtp_static"]
_good = _cp.deepcopy(A["fp8_mtp2"])
def _ad(**mut):
    a = _cp.deepcopy(_good)
    for k, v in mut.items():
        if v is _DEL:
            a.pop(k, None)
        else:
            a[k] = v
    return _cl(a)["adoptable"]
_DEL = object()
_lvl = _good.get("level_wanted") or 16
_neg = {"base adoptable": _ad(), "missing XL": not _ad(xl_gate=None), "XL INCOMPLETE": not _ad(xl_gate="INCOMPLETE"),
        "no canary": not _ad(canary=_DEL), "canary errors": not _ad(canary={"level": _lvl, "errors": 1, "foreign_leaks": 0, "own_hits": 3}),
        "canary leak": not _ad(canary={"level": _lvl, "errors": 0, "foreign_leaks": 1, "own_hits": 3}),
        "canary other level": not _ad(canary={"level": 12 if _lvl != 12 else 16, "errors": 0, "foreign_leaks": 0, "own_hits": 3}),
        "DP diagnostic": not _ad(diagnostic_only=True),
        "PCIe unverified": not _ad(env={"VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "1"}, pcie_ipc={"initialized": False})}
_w = _cp.deepcopy(_good["short"]); _w1 = dict(_w, no_first_token=1)
_neg["window healthy"] = _wh(_w, _lvl); _neg["window no_first_token=1 unhealthy"] = not _wh(_w1, _lvl)
_g30 = chr(10).join(" ".join("0" for _ in range(30)) for _ in range(30))
_neg["text 0*250 bad"] = _txb("0" * 250); _neg["text 0+!+0 bad"] = _txb("0" * 40 + "!" * 60 + "0" * 60)
_neg["text 0 0 0 x200 bad"] = _txb(" ".join("0" * 200)); _neg["text grids ok"] = not any(_txb(x) for x in ("[[0,0],[0,0]]", "0 0 1" + chr(10) + "0 0 0", _g30))
_neg["text prose ok"] = not _txb("Let me look at the first example carefully. The input grid has a red line. " * 6)
_one = {k: v[:1] for k, v in G["VMS_REF"].items() if k in ("serial_a", "serial_b", "conc_a", "conc_b")}
_neg["judge 1 row INCOMPLETE"] = G["vms_judge"](_one["serial_a"], _one["conc_a"], _one)[0] == "INCOMPLETE"
def _st(keys, ign):
    d = Path(_tf.mkdtemp())
    (d / "config.json").write_text(json.dumps({"quantization_config": {"ignore": ign, "format": "pack-quantized"}}), encoding="utf-8")
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "x" for k in keys}}), encoding="utf-8")
    return bool(_ist(d))      # vms_int4_mtp_static returns the ok flag
_saved = G["VMS"].get("int4_mtp_static")
_neg["static dense+ignored ok"] = _st(["mtp.fc.weight", "mtp.layers.0.mlp.gate_proj.weight"], ["re:^mtp.*"])
_neg["static dense not ignored FAIL"] = not _st(["mtp.fc.weight", "mtp.layers.0.mlp.gate_proj.weight"], [])
_neg["static packed+ignored FAIL"] = not _st(["mtp.fc.weight_packed", "mtp.fc.weight_scale"], ["re:^mtp.*"])
_neg["static packed not ignored ok"] = _st(["mtp.fc.weight_packed", "mtp.fc.weight_scale"], [])
G["VMS"]["int4_mtp_static"] = _saved
check("rev 2g negative tests (GPT rev 2f audit R1-R5 / R2F-01..10): fail-closed classifier, first-token coverage, "
      "structural grid rule, planned row counts, INT4 MTP static check", all(_neg.values()), {k: v for k, v in _neg.items() if not v})
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
      _rows and all(r["tokens"] and len(r["margins"]) == len(r["tokens"]) for x in _rows for v in x["rows"].values() for r in (v or []))
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
