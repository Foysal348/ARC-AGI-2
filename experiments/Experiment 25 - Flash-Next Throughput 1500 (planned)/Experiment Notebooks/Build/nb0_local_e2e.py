"""E25 NB0 local end-to-end runs: the exported notebook code (all cells, all 5 boots, RC2's merlin_boot / with_servers /
oracle path) against nb0_fake_server.py (E23_FAKE_SERVER mode), once per fault mode, each asserting the verdict and the
decision-table findings the mode must produce.

Usage: python nb0_local_e2e.py <fake input dir> <work dir> [mode ...]   (default modes: all)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INP, WORK = Path(sys.argv[1]), Path(sys.argv[2])
EXPECT = {   # mode -> (CD triage, CE triage, findings that must appear, findings that must not)
    "clean": ("PENDING_LONG_ARC_ACCURACY", "PENDING_LONG_ARC_ACCURACY", {"KV_CLEAN", "RC2_ANCHOR"},
              {"KV_DIVERGENCE_FROM_2K", "MTP_ONLY", "NATIVE_SM89_E4M3_PATH_DEFECT", "REPEAT_DIVERGENCE", "DECODE_VS_PROMPT_LOGPROB_MISMATCH"}),
    "native": ("NO_GO_FP8_TF_DIVERGENCE", "PENDING_LONG_ARC_ACCURACY",
               {"KV_DIVERGENCE_FROM_2K", "NATIVE_SM89_E4M3_PATH_DEFECT", "READ_PATH_CHANGES_LOGITS"}, {"MTP_ONLY"}),
    "kv_long": ("NO_GO_FP8_TF_DIVERGENCE", "NO_GO_FP8_TF_DIVERGENCE", {"KV_DIVERGENCE_LONG_ONLY", "FP8_DIVERGENCE_NOT_THE_NATIVE_CAST"},
                {"KV_DIVERGENCE_FROM_2K", "NATIVE_SM89_E4M3_PATH_DEFECT"}),
    "mtp": ("NO_GO_FP8_TF_DIVERGENCE", "NO_GO_FP8_TF_DIVERGENCE",
            {"MTP_ONLY", "MTP_CHANGES_TARGET_LOGITS_FP8", "MTP_GREEDY_DIVERGENCE_HIGH_MARGIN"}, {"NATIVE_SM89_E4M3_PATH_DEFECT"}),
    "path": (None, None, {"DECODE_VS_PROMPT_LOGPROB_MISMATCH"}, set()),
    "shift": (None, None, {"POSITION_SHIFT_ARTEFACT"}, set()),
    "repeat": (None, None, {"REPEAT_DIVERGENCE"}, set()),
}
modes = sys.argv[3:] or list(EXPECT)
ok_all = True
for mode in modes:
    d = WORK / mode
    if d.exists():
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    for x in ("out", "tmp", "shm", "proc", "cgroup"):
        (d / x).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "E23_FAKE_SERVER": str(HERE / "nb0_fake_server.py"), "E23_OUT": str(d / "out"), "E23_TMP": str(d / "tmp"),
           "E23_INPUT": str(INP), "RC2_SUBMISSION": str(d / "out" / "submission.json"), "RC2_SHM": str(d / "shm"),
           "NB0_BUDGET_SEC": "3600", "NB0_PROC": str(d / "proc"), "NB0_CGROUP": str(d / "cgroup"), "FAKE_TF_MODE": mode,
           "FAKE_BOOT_SEC": "1", "NB0_PORT_BASE": os.environ.get("NB0_PORT_BASE", "18710")}
    env.pop("NB0_UNIT", None)
    if os.environ.get("E2E_REAL_PROC") == "1":       # Linux: the real /proc (session scan, reap, load monitor on real pids)
        env.pop("NB0_PROC", None)
    env.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    t = time.time()
    with open(d / "run.log", "w", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, str(HERE / "nb0_code_export.py")], env=env, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(d)).returncode
    sec = time.time() - t
    res = []

    def check(name, cond, detail=""):
        res.append((name, bool(cond), detail))
        print(f"  {'PASS' if cond else 'FAIL'} [{mode}] {name}" + (f"  [{detail}]" if detail else ""))

    print(f"== mode {mode}: rc {rc}, {sec:.0f} s")
    try:
        R = json.loads((d / "out" / "nb0_results.json").read_text(encoding="utf-8"))
        V = json.loads((d / "out" / "nb0_verdict.json").read_text(encoding="utf-8"))
    except Exception as e:
        check("results written", False, repr(e))
        ok_all = False
        continue
    check("exit code 0", rc == 0)
    st = V["boots"]
    check("all 5 boots DONE", all(v == "DONE" for v in st.values()), json.dumps(st))
    sec_ = R["sections"].get("NB0") or {}
    check("all 5 boots CLEAN (probes / sampled / 4-row oracle complete)", all(r.get("correctness") == "CLEAN" for r in sec_.values()),
          json.dumps({k: r.get("correctness") for k, r in sec_.items()}))
    check("config_ok on every boot (kv / seqs / MTP / byte marker / prefix caching from the server log)",
          all((r.get("boot_facts") or {}).get("config_ok") for r in sec_.values()))
    an = R["analysis"]
    cds = {b: e["last256"].get("status") for b, e in an["pairs"]["CD"]["bands"].items()}
    check("C-D last-256 MEASURED in all 4 bands", all(v == "MEASURED" for v in cds.values()), json.dumps(cds))
    ct = [an["per_boot"].get(k, {}).get("cached_tokens", {"missing": None}) for k in "DCEBA"]
    check("teacher-forced requests never read the prefix cache (cached_tokens 0)", all(x == [0, 0] for c in ct for x in c.values()), json.dumps(ct)[:200])
    scc = [((an["per_boot"].get(k) or {}).get("sc", {}).get(b) or {}).get("cached_tokens", [None, None]) for k in "DCEBA" for b in ("2k", "8k", "40k")]
    check("decode repeat hits the prefix cache (cold 0, cached > 0 at 8k / 40k)", all(c[0] == 0 for c in scc) and all((c[1] or 0) > 0 for c in scc[1::3] + scc[2::3]) and all(c[1] == 0 for c in scc[0::3]),
          json.dumps(scc))
    raw = list((d / "out" / "tf_raw").glob("*.npz"))
    check("raw arrays on disk (5 boots x (8 TF + 3 SC-TF + 6 decode))", len(raw) == 5 * 17, len(raw))
    req = (d / "out" / "traces" / "nb0_requests.jsonl").read_text(encoding="utf-8").splitlines()
    check("every NB0 request logged", len(req) == 5 * 17, len(req))
    plans = R["setup"]["nb0_plans"]
    check("boot B / A argv without --speculative-config, D / C / E with it",
          all(("--speculative-config" in p["argv_tail"]) == (p["boot"] in "DCE") for p in plans.values()))
    lm = all((r.get("load_monitor") or {}).get("samples", 0) >= 0 for r in sec_.values())
    check("load monitor ran for every boot", lm and all(r.get("load_monitor") is not None for r in sec_.values()))
    if os.environ.get("E2E_REAL_PROC") == "1" and Path("/proc/self/stat").exists():
        per = [len((r.get("load_monitor") or {}).get("per_process") or {}) for r in sec_.values()]
        reaps = [((r.get("boot_gate") or {}).get("host_prep") or {}).get("procs", {}).get("ok") for r in sec_.values()]
        check("real /proc: the load monitor saw the fake server session (server + 4 workers) on every boot", all(n >= 5 for n in per), per)
        check("real /proc: every pre-boot reap ok", all(x is True for x in reaps), reaps)
    found = {f["finding"] for f in V["findings"]}
    cd_t, ce_t, must, mustnt = EXPECT[mode]
    if cd_t:
        check(f"verdict FP8 native (C-D) = {cd_t}", V["gates"]["fp8_native_production"] == cd_t, V["gates"]["fp8_native_production"])
        check(f"verdict FP8 byte decoder (C-E) = {ce_t}", V["gates"]["fp8_byte_decoder"] == ce_t, V["gates"]["fp8_byte_decoder"])
    check(f"findings include {sorted(must)}", must <= found, sorted(found))
    check(f"findings exclude {sorted(mustnt)}", not (mustnt & found), sorted(found & mustnt))
    gv = [V["gates"][k] for k in ("fp8_native_production", "fp8_byte_decoder")] + [V["gates"][pk][x] for pk in ("CD", "AB", "CE") for x in ("screening", "triage")]
    check("no gate value is GO", not any(g == "GO" or g.startswith("GO_") for g in gv), gv)
    print("  gates:", json.dumps({k: V["gates"][k] for k in ("fp8_native_production", "fp8_byte_decoder")}), "| findings:", sorted(found))
    ok_all &= all(r[1] for r in res)
print("E2E ALL PASS" if ok_all else "E2E FAILURES")
sys.exit(0 if ok_all else 1)
