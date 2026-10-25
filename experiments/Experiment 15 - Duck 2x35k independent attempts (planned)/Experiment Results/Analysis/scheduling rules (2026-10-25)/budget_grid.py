"""Thinking-budget grid at hidden scale (2026-10-25): credit at 11.6 h and time needed, per (attempt 1, attempt 2) budget.
Same replay as time_slice_sim.py (240 synthetic tasks from the two XHIGH EVAL120 runs, 16 streams, 36.6 s per 1k)."""
import io, contextlib, statistics as st
with contextlib.redirect_stdout(io.StringIO()):
    import time_slice_sim as S
acc = sorted(x["acc_tok"] for _, a, b in S.TASKS for x in (b,) if any(x["acc_q"]) and x["acc_tok"])
frac = lambda c: sum(1 for v in acc if v <= c) / len(acc)
need = lambda c1, c2: sum(min(a["tokens"], c1) + min(b["tokens"], c2) for _, a, b in S.TASKS) * S.SEC_PER_K / 1000 / S.STREAMS / 3600
B1S = (35000, 40000, 45000, 50000, 55000, 60000, 70000)
B2S = (30000, 35000, 45000, 50000, 60000)
print("### P0 (now: attempt 2 on every task) - task credit at 11.6 h (of 240) | hours both attempts need")
print("| attempt 1 \ attempt 2 | " + " | ".join(f"{b // 1000}k" for b in B2S) + " |")
print("|---" * (len(B2S) + 1) + "|")
for c1 in B1S:
    cells = []
    for c2 in B2S:
        if c2 > c1:
            cells.append("-"); continue
        v = S.simulate("P0", c1, c2)[-1]
        cells.append(f"{v:.1f} ({need(c1, c2):.1f} h)")
    print(f"| {c1 // 1000}k | " + " | ".join(cells) + " |")
print("\n### R + stage 3 (estimate; attempt-3 p = 0.10 uncapped x share of solves within the attempt-2 budget)")
print("| attempt 1 \ attempt 2 | " + " | ".join(f"{b // 1000}k" for b in B2S) + " |")
print("|---" * (len(B2S) + 1) + "|")
for c1 in B1S:
    cells = []
    for c2 in B2S:
        if c2 > c1:
            cells.append("-"); continue
        p = 0.10 * frac(c2)
        vals = [S.simulate("R", c1, c2, p, seed=s)[-1] for s in range(60)]
        cells.append(f"{st.mean(vals):.1f}")
    print(f"| {c1 // 1000}k | " + " | ".join(cells) + " |")
print("\nshare of correct solves within the budget (G4B XHIGH b0, 69 solves):",
      {f"{c // 1000}k": round(frac(c), 3) for c in (30000, 35000, 40000, 45000, 50000, 55000, 60000)})
