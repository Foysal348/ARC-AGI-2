import sys, json, csv, collections, io, contextlib, importlib.util
with contextlib.redirect_stdout(io.StringIO()):
    from load_attempts import load, SOL
    spec = importlib.util.spec_from_file_location("BL", r"D:\ARC AGI 2\resources\Puzzle Capability Ledger\build_ledger.py")
    BL = importlib.util.module_from_spec(spec); spec.loader.exec_module(BL)
rows = load()
R = collections.defaultdict(dict)
for r in rows: R[(r["run"], r["task"])][r["b"]] = r
CH = json.load(open(BL.FX / "arc-agi_evaluation_challenges.json"))
e10 = [x for x in BL.RUNS if x[0].startswith("E10 ORDER2")][0]
d10 = {k: v["hit"] for k, v in BL.load_run(e10[3], e10[4], CH).items()}
ap = json.load(open(r"D:\ARC AGI 2\resources\Puzzle Capability Ledger\arcprize_qwen38_27b_public_eval.json"))["tasks"]
for run in ("G4B_XHIGH_E16", "G4B_XHIGH_noE16"):
    other = "G4B_XHIGH_noE16" if run == "G4B_XHIGH_E16" else "G4B_XHIGH_E16"
    noacc = [t for t in SOL if 0 in R.get((run, t), {}) and R[(run, t)][0]["cls"] < 2]
    tn = lambda f: sum(sum(bool(f(t, q)) for q in range(len(SOL[t]))) / len(SOL[t]) for t in noacc)
    duck2 = tn(lambda t, q: 0 in R.get((other, t), {}) and R[(other, t)][0]["acc_q"][q])
    print(f"{run}: {len(noacc)} tasks with no accepted answer in attempt 1 | per-attempt credit on them: "
          f"E10 direct 86k {tn(lambda t, q: d10.get((t, q), 0)) / len(noacc):.3f}, "
          f"ARC Prize xhigh direct {sum(ap[t]['xhigh'] for t in noacc) / len(noacc):.3f}, "
          f"a fresh Duck xhigh attempt (other run) {duck2 / len(noacc):.3f}")
# tokens of the E10 direct run per task on those tasks
import statistics as st
recs = [json.loads(l) for l in open(e10[4], encoding="utf-8")]
tok = collections.defaultdict(int)
for r in recs:
    if r.get("resample_of"): continue
    tok[r["task_id"]] += int(r.get("think_tokens") or 0) + int(r.get("answer_tokens") or 0)
noacc = [t for t in SOL if 0 in R.get(("G4B_XHIGH_E16", t), {}) and R[("G4B_XHIGH_E16", t)][0]["cls"] < 2]
dt = [tok[t] for t in noacc if t in tok]
fresh = [R[("G4B_XHIGH_noE16", t)][0]["tokens"] for t in noacc]
print("E10 direct tokens per task (those tasks): mean", round(st.mean(dt)), "| fresh Duck attempt tokens: mean", round(st.mean(fresh)))
print("credit per 100k tokens: direct", round(100000 * 0.255 / st.mean(dt), 3), "| Duck", round(100000 * 0.132 / st.mean(fresh), 3))
print("keys in a record:", sorted(recs[0].keys())[:40])
