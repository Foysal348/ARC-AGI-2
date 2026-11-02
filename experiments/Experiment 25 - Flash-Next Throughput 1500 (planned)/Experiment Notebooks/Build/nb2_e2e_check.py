"""E25 NB2 local fake-server end-to-end: runs nb2_code_export.py against nb2_fake_server.py (the real Merlin 3.0.0
launch.build() argv, NB2_MERLIN_SRC) on a fake input tree, then checks the output (fake F = 4 + K, pool 149 blocks).
FAKE mode only: NB2_FAKE_HOST_FLOOR_MIB lowers the host RAM floor (the Kaggle CPU kit has a 30 GiB cgroup; KIT-1);
NB2_OPTIONAL=K0,CTRL so every boot's code path runs (the notebook default is A / B / C only). A boot that did not reach
DONE gets FAIL lines, never a KeyError (KIT-2).

Usage: python nb2_e2e_check.py <fake input dir> <work dir> [<merlin src dir>] [--check-only]
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INP, W = Path(sys.argv[1]), Path(sys.argv[2])
MSRC = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src"
if "--check-only" not in sys.argv:
    shutil.rmtree(W, ignore_errors=True)
    for d in ("out", "tmp", "shm"):
        (W / d).mkdir(parents=True)
    env = dict(os.environ, E23_FAKE_SERVER=str(HERE / "nb2_fake_server.py"), NB2_MERLIN_SRC=MSRC, FAKE_SPEED="300",
               E23_OUT=str(W / "out"), E23_TMP=str(W / "tmp"), E23_INPUT=str(INP), NB2_SUBMISSION=str(W / "out/submission.json"),
               NB2_SHM=str(W / "shm"), NB2_BUDGET_SEC="5400", E23_WIN_SEC="5", E23_EST_SCALE="0.01", NB2_DECAY_MIN="0.6",
               NB2_DECAY_SUB_MIN="0.25", NB2_FAKE_EOS_AFTER="6000", PYTHONIOENCODING="utf-8",
               NB2_FAKE_HOST_FLOOR_MIB=os.environ.get("NB2_FAKE_HOST_FLOOR_MIB") or "1024",
               NB2_OPTIONAL=os.environ.get("NB2_OPTIONAL", "K0,CTRL"))
    env.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    with open(W / "run.log", "w", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, str(HERE / "nb2_code_export.py")], env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    print("notebook rc", rc)

R = json.loads((W / "out" / "nb2_results.json").read_text())
S2 = R["sections"]["S2"]
res = []


def check(n, c, d=""):
    res.append(bool(c))
    print(("PASS " if c else "FAIL ") + n + (f"  [{str(d)[:260]}]" if d else ""))


def boot_checks(name, r):
    key, k = r["cfg"]["key"], r["cfg"]["k"]
    check(f"{name}: config read back from the server log", (r.get("config_readback") or {}).get("ok") is True, r.get("config_readback"))
    fc = r["f_census"]
    check(f"{name}: F census = fake F {4 + k}, pool 149", fc.get("status") == "MEASURED" and fc["F_int"] == 4 + k
          and abs(fc["pool_blocks"] - 149) < 1, {x: fc.get(x) for x in ("F", "pool_blocks", "idle")})
    check(f"{name}: idle census: reset_prefix_cache " + ("ok" if not r["cfg"]["exact_rc2"] else "absent (CTRL, no dev mode)"),
          fc["idle"]["reset_prefix_cache"]["ok"] is (not r["cfg"]["exact_rc2"]), fc["idle"])
    st = {kk: w for kk, w in r["steady"].items()}
    ran = {kk: w for kk, w in st.items() if w.get("status") == "STEADY"}
    check(f"{name}: windows STEADY with real running == level, tick/cycle, all prefilled", ran and all(
        w["real_running_eq_level"] and w["tick_ms"] and w["cycle_ms"] == round(2 * w["tick_ms"], 1) and w["all_prefilled_at_start"]
        for w in ran.values()), {kk: w.get("status") for kk, w in st.items()})
    if k:
        check(f"{name}: per-position acceptance has {k} positions in every window", all(
            len((w.get("per_pos") or {}).get("per_position") or []) == k for w in ran.values()),
            {kk: (w.get("per_pos") or {}).get("per_position") for kk, w in ran.items()})
    if r["cfg"]["seqs"] > 8:
        hi = [kk for kk in ran if int(kk.split("@")[1]) > 8]
        check(f"{name}: > 8-stream windows ran at 8k (or were substituted / skipped with a reason)", hi or any(
            "substituted_level" in w or w.get("status", "").startswith("SKIPPED") for kk, w in st.items() if int(kk.split("@")[1]) > 8), st.keys())
    check(f"{name}: exact diff recorded", "flags_changed" in (r["cfg"].get("exact_diff") or {}), r["cfg"].get("diff_vs_rc2_fp8"))
    if k and not r["cfg"]["exact_rc2"] and r["cfg"].get("long_window"):
        check(f"{name}: decay windows + cohort replays + verdicts", len(r["decay"]["windows"]) >= 2 and len(r["cohort"]) >= 2
              and r["cohort_verdict"]["status"] in ("STABLE", "UPTIME_DECAY_SUSPECT"), [r["decay"].get("status"), r["cohort_verdict"]])
    elif k and not r["cfg"]["exact_rc2"]:
        check(f"{name}: no long window (boot A only) + cohort start / after ladder", r["decay"].get("status") == "N/A_NOT_THE_LONG_WINDOW_BOOT"
              and len(r["cohort"]) >= 2, [r["decay"].get("status"), len(r.get("cohort") or [])])
    if not r["cfg"]["exact_rc2"]:
        nb = (r.get("trace") or {}).get("nb2") or {}
        check(f"{name}: trace: 4 ranks, PP2 overlap, verify rows {k + 1} tokens/req", len(nb.get("ranks") or {}) == 4 and "both_busy_pct" in (nb.get("pp2") or {})
              and nb["verify_rows"]["tokens_per_req"] == [float(k + 1)], [nb.get("pp2"), nb.get("verify_rows")])
    check(f"{name}: step model fitted", (r.get("step_model") or {}).get("status") in ("FITTED", "INCONCLUSIVE_FIT"), r.get("step_model"))
    check(f"{name}: warm-up clean", (r.get("warmup") or {}).get("clean") is True, r.get("warmup"))
    check(f"{name}: boot gate recorded page cache + host gate", "page_cache" in r["boot_gate"] and "host_gate" in r["boot_gate"])


for name, r in S2.items():
    if not isinstance(r, dict) or not r.get("cfg"):
        continue
    check(f"{name}: DONE / CLEAN", r.get("status") == "DONE" and r.get("correctness") == "CLEAN",
          [r.get("status"), r.get("correctness"), r.get("early_stop")])
    if r.get("status") != "DONE":                  # KIT-2: the per-boot facts below exist only for a boot that reached DONE
        check(f"{name}: boot did not reach DONE, per-boot checks not possible", False, [r.get("status"), r.get("boot_gate", {}).get("ok")
                                                                                     if isinstance(r.get("boot_gate"), dict) else None])
        continue
    try:
        boot_checks(name, r)
    except Exception as e:
        check(f"{name}: per-boot checks raised {type(e).__name__}: {e}"[:200], False)
for name, r in S2.items():
    for kk, w in (r.get("steady") or {}).items():
        if w.get("status") == "STEADY":
            check(f"{name} {kk}: tick / cycle / completed micro-batch rate / wall tok/s / phase / a per verify token reported",
                  all(w.get(x) is not None for x in ("tick_ms", "cycle_ms", "completed_microbatch_rate_per_s", "wall_aggregate_tok_s",
                                                     "phase", "accepted_per_verify_token", "microbatch")) and w["microbatch"]["covered"])
check("source hashes recorded", len(R.get("sources") or {}) >= 7, R.get("sources"))
sc = json.loads((W / "out" / "lever_scorecard.json").read_text())
keys = {r["cfg"]["key"] for r in S2.values() if isinstance(r, dict) and r.get("cfg")}
n_rows = ("A" in keys) + len(keys & {"B", "C", "BH", "CH", "K0"}) + (1 if keys else 0)
check("scorecard rows: D2 (A) + one D1 row per K boot + D3", len(sc["rows"]) == n_rows and n_rows >= 3, [x["lever"] for x in sc["rows"]])
reqs = [json.loads(l) for l in (W / "out" / "traces" / "requests.jsonl").read_text().splitlines()]
check("every request saved with its full text", reqs and all("text" in q and "tokens" in q for q in reqs), len(reqs))
check("submission written (120 tasks)", len(json.loads((W / "out" / "submission.json").read_text())) == 120)
print(f"\n{sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
