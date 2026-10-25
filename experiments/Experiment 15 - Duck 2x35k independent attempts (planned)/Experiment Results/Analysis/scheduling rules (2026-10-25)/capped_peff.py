import statistics as st, time
import time_slice_sim as S
t=time.time(); SL=S.SLICES
print("| policy (CAPPED 45K30K) | " + " | ".join(f"{h} h" for h in SL) + " |"); print("|---"*(len(SL)+1)+"|")
for pol in ("P0", "R"):
    v = S.simulate(pol, 45000, 30000, None); print(f"| {pol} | " + " | ".join(f"{x:.1f}" for x in v) + " |")
for pol in ("P0", "R"):
    for p in (0.043, 0.068, 0.10):
        vals = [S.simulate(pol, 45000, 30000, p, seed=s) for s in range(100)]
        m = [st.mean(v[j] for v in vals) for j in range(len(SL))]
        print(f"| {pol}+3, p {p} | " + " | ".join(f"{x:.1f}" for x in m) + " |")
print("sec", round(time.time()-t,1))
