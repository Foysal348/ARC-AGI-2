"""Replay of the policy "attempt 2 only where attempt 1 has no accepted answer; the saved tokens buy attempt 3 on tasks
still unsolved" on our real runs, at EQUAL generated tokens (2026-10-25, user: apply it to attempt 2 in our runs).
P0 = current (attempt 2 on every task). P1 = skip attempt 2 on accepted tasks + attempt 3 from the saved tokens.
P2 = P0 + attempt 3 with the same extra tokens on top (what spare time allows without skipping)."""
import collections, random, statistics as st
from load_attempts import load, SOL

rows = load()
R = collections.defaultdict(dict)
for r in rows:
    R[(r["run"], r["task"])][r["b"]] = r
credit = lambda q: sum(q) / len(q)
union = lambda *qs: [any(x) for x in zip(*qs)]
XH = ("G4B_XHIGH_E16", "G4B_XHIGH_noE16", "E15B_45K30K", "E15B_40K35K")


def pool_for(t, used):
    """Independent xhigh attempts on task t that the replayed run did not use (attempt-3 candidates)."""
    out = []
    for run in XH:
        d = R.get((run, t), {})
        for b, x in d.items():
            if run.startswith("G4B") and b == 1:
                continue                      # the G4B b1 used the indexed view (0 credit): not a fresh facts attempt
            if id(x) not in used:
                out.append(x)
    return out


def replay(name, a1, a2, shuffles=500, seed=0):
    """a1[t], a2[t]: the run's attempt 1 and attempt 2 per task (dicts from load())."""
    rng = random.Random(seed)
    T = sorted(a1)
    used = {id(a1[t]) for t in T} | {id(a2[t]) for t in T if a2.get(t)}
    p0 = {t: union(a1[t]["acc_q"], a2[t]["acc_q"]) if a2.get(t) else a1[t]["acc_q"] for t in T}
    tok0 = sum(a1[t]["tokens"] + (a2[t]["tokens"] if a2.get(t) else 0) for t in T)
    skip = [t for t in T if a1[t]["cls"] >= 2 and a2.get(t)]
    saved = sum(a2[t]["tokens"] for t in skip)
    p1base = {t: (a1[t]["acc_q"] if t in skip else p0[t]) for t in T}
    lost = sum(credit(p0[t]) - credit(p1base[t]) for t in skip)
    lost_tasks = [t for t in skip if credit(p0[t]) > credit(p1base[t])]

    def attempt3(base, budget):
        g_all, n_all = [], []
        for _ in range(shuffles):
            cur = {t: list(v) for t, v in base.items()}
            left = budget
            order = [t for t in T if not any(cur[t])]        # still unsolved, in the run's task order
            pools = {t: rng.sample(pool_for(t, used), len(pool_for(t, used))) for t in order}
            n = 0
            progress = True
            while progress:                                  # round robin: one more attempt per unsolved task
                progress = False
                for t in order:
                    if any(cur[t]) or not pools[t]:
                        continue
                    x = pools[t].pop()
                    if x["tokens"] > left:
                        continue
                    left -= x["tokens"]; n += 1; progress = True
                    cur[t] = union(cur[t], x["acc_q"])
            g_all.append(sum(credit(v) for v in cur.values())); n_all.append(n)
        return st.mean(g_all), st.mean(n_all)

    c0 = sum(credit(v) for v in p0.values())
    c1, n1 = attempt3(p1base, saved)
    c2, n2 = attempt3(p0, saved)
    print(f"| {name} | {len(T)} | {c0:.2f} | {len(skip)} | {saved:,} | {lost:.2f} ({', '.join(lost_tasks) or '-'}) | "
          f"{n1:.1f} | **{c1:.2f}** ({c1 - c0:+.2f}) | {c2:.2f} ({c2 - c0:+.2f}) |")


print("| replayed run | tasks | P0 credit (now) | attempt 2s skipped | tokens saved | credit lost by skipping | "
      "attempt 3s bought | **P1 credit** (vs P0) | P2 = P0 + same tokens for attempt 3 (vs P0) |")
print("|---|---|---|---|---|---|---|---|---|")
for run in ("E15B_45K30K", "E15B_40K35K"):
    a1 = {t: R[(run, t)][0] for (r, t) in R if r == run}
    a2 = {t: R[(run, t)].get(1) for (r, t) in R if r == run}
    replay(run + " (b0, b1 = fresh)", a1, a2)
for ra, rb in (("G4B_XHIGH_E16", "G4B_XHIGH_noE16"), ("G4B_XHIGH_noE16", "G4B_XHIGH_E16")):
    a1 = {t: R[(ra, t)][0] for t in SOL if 0 in R.get((ra, t), {}) and 0 in R.get((rb, t), {})}
    a2 = {t: R[(rb, t)][0] for t in a1}
    replay(f"EVAL120: a1 = {ra} b0, a2 = {rb} b0", a1, a2)
print("\nAttempt 3 = an unused independent xhigh attempt on the same task (G4B XHIGH b0 of either run, E15B b0/b1 of either "
      "run). For EVAL120 only the 20 E15B tasks have such attempts left, so P1/P2 there can buy attempt 3 only on those "
      "20 tasks (a lower bound for the other 100).")
