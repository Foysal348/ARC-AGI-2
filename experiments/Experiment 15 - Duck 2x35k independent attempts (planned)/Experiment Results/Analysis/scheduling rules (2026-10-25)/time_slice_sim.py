"""Time-sliced scheduler replay at hidden scale (2026-10-25, user: apply "a new attempt only where no accepted answer
yet" to attempt 2 and show the solve count by time slice).

240 synthetic tasks = the 120 EVAL120 tasks x 2 directions: attempt 1 = b0 of one XHIGH G4B run, attempt 2 = b0 of the other
(a real, independent, fresh xhigh attempt). 16 streams, 36.6 stream-seconds per 1k generated tokens (measured, E15B 45K30K),
solve window 11.6 h. A correct answer counts at the moment its first accepted candidate appeared (start + acc tokens).
Stage 1 = every attempt 1 in ORDER2 + cells order; stage 2 starts only after every attempt 1 has started, by class.
Policies:
  P0 = now (attempt 2 on every task, class order 0, 1, 2 flagged first, 3)
  R  = attempt 2 only where attempt 1 has no accepted answer (the rule applied to attempt 2)
  P0+3 / R+3 = the same, then attempt 3 on tasks with no accepted answer until the window ends (ESTIMATE: Bernoulli p,
       timing / tokens from the real fresh attempts; p = 0.10 / 0.156 from Notes/12 A5)
Scenarios: raw = the attempts' real token counts; capped = 45K30K budgets (an attempt 1 that accepted after 45k, or an
attempt 2 / 3 after 30k, counts as no accept and costs the budget)."""
import collections, heapq, random, statistics as st
from load_attempts import load, SOL, CH

rows = load()
R = collections.defaultdict(dict)
for r in rows:
    R[(r["run"], r["task"])][r["b"]] = r
SEC_PER_K = 36.6
WINDOW = 11.6 * 3600
STREAMS = 16
MIN_ADMIT = 300


def cells(g):
    return len(g) * len(g[0]) if g and g[0] else 0


def okey(t):
    c = CH[t]
    s = int(len(c["train"]) >= 3) + int(len(c["test"]) == 1)
    tc = sum(cells(p["input"]) + cells(p["output"]) for p in c["train"]) + sum(cells(p["input"]) for p in c["test"])
    return (-s, tc, t)


def att(x, cap):
    """(tokens used, tokens at first correct accept or None, per-query correct list, class)"""
    tok = x["tokens"] if cap is None else min(x["tokens"], cap)
    ok = any(x["acc_q"]) and x["acc_tok"] is not None and (cap is None or x["acc_tok"] <= cap)
    acc_any = x["acc_tok"] is not None and (cap is None or x["acc_tok"] <= cap)     # any accepted (right or wrong)
    cls = x["cls"] if (cap is None or acc_any or x["cls"] < 2) else (0 if x["prog"] else 1)
    return tok, (x["acc_tok"] if ok else None), (x["acc_q"] if ok else [False] * len(x["acc_q"])), cls, x["flagged"]


TASKS = []
for ra, rb in (("G4B_XHIGH_E16", "G4B_XHIGH_noE16"), ("G4B_XHIGH_noE16", "G4B_XHIGH_E16")):
    for t in sorted(SOL, key=okey):
        if 0 in R.get((ra, t), {}) and 0 in R.get((rb, t), {}):
            TASKS.append((t, R[(ra, t)][0], R[(rb, t)][0]))
TASKS.sort(key=lambda x: okey(x[0]))
fresh = [b for _, a, b in TASKS]


def simulate(policy, cap1, cap2, p3=None, seed=0):
    rng = random.Random(seed)
    solved_tok = [x["acc_tok"] for x in fresh if any(x["acc_q"]) and x["acc_tok"] and (cap2 is None or x["acc_tok"] <= cap2)]
    fail_tok = [min(x["tokens"], cap2) if cap2 else x["tokens"] for x in fresh if not any(x["acc_q"])]
    n = len(TASKS)
    hit = [[False] * len(SOL[t]) for t, _, _ in TASKS]
    has_acc = [False] * n
    events = []                     # (time, task index, per-query list) credit events
    free = [0.0] * STREAMS          # heap of times when a stream frees up
    heapq.heapify(free)
    a1_end = {}
    queue2 = []                     # (ready time, prio, i)
    starts = 0

    def run(i, x, cap, t0):
        tok, acc_t, q, cls, flag = att(x, cap)
        end = t0 + tok * SEC_PER_K / 1000
        if acc_t is not None and t0 + acc_t * SEC_PER_K / 1000 <= WINDOW:
            events.append((t0 + acc_t * SEC_PER_K / 1000, i, q))
        return end, cls, flag, (x["acc_tok"] is not None and (cap is None or x["acc_tok"] <= cap))

    # stage 1
    last_a1_start = 0.0
    for i, (t, a, b) in enumerate(TASKS):
        t0 = heapq.heappop(free)
        if WINDOW - t0 < MIN_ADMIT:
            heapq.heappush(free, t0); break
        end, cls, flag, acc = run(i, a, cap1, t0)
        has_acc[i] = acc
        a1_end[i] = (end, cls, flag)
        last_a1_start = t0
        heapq.heappush(free, min(end, WINDOW))
    # stage 2: items become available at their attempt-1 end, but none starts before the last attempt 1 started
    items = []
    for i, (end, cls, flag) in a1_end.items():
        if policy.startswith("R") and has_acc[i]:
            continue
        items.append((max(end, last_a1_start), (cls, 0 if flag else 1, i), i))
    items.sort()
    pending = []
    k = 0
    while True:
        t0 = heapq.heappop(free)
        while k < len(items) and items[k][0] <= t0:
            heapq.heappush(pending, (items[k][1], items[k][2])); k += 1
        if not pending:
            if k < len(items):          # nothing ready: the stream waits for the next item
                heapq.heappush(free, items[k][0]); continue
            heapq.heappush(free, t0); break
        if WINDOW - t0 < MIN_ADMIT:
            break
        _, i = heapq.heappop(pending)
        end, cls, flag, acc = run(i, TASKS[i][2], cap2, t0)
        has_acc[i] = has_acc[i] or acc
        heapq.heappush(free, min(end, WINDOW))
    # stage 3 (estimate)
    if p3 is not None:
        un = [i for i in range(n) if not has_acc[i]]
        rnd = 0
        while un:
            progressed = False
            for i in list(un):
                t0 = heapq.heappop(free)
                if WINDOW - t0 < MIN_ADMIT:
                    heapq.heappush(free, t0); un = []; break
                if rng.random() < p3:
                    acc_t = rng.choice(solved_tok)
                    tc = t0 + acc_t * SEC_PER_K / 1000
                    if tc <= WINDOW:
                        events.append((tc, i, [True] * len(SOL[TASKS[i][0]])))
                    end = tc
                    un.remove(i)
                else:
                    end = t0 + rng.choice(fail_tok) * SEC_PER_K / 1000
                heapq.heappush(free, min(end, WINDOW))
                progressed = True
            rnd += 1
            if not progressed or rnd > 6:
                break
    events.sort()
    out = []
    ei = 0
    cur = [[False] * len(SOL[t]) for t, _, _ in TASKS]
    for h in SLICES:
        while ei < len(events) and events[ei][0] <= h * 3600:
            _, i, q = events[ei]
            cur[i] = [a or b for a, b in zip(cur[i], q)]; ei += 1
        out.append(sum(sum(v) / len(v) for v in cur))
    return out


SLICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11.6]


def main():
    for name, c1, c2 in (("RAW (real token counts of the G4B attempts)", None, None), ("CAPPED to 45K30K budgets", 45000, 30000)):
        print(f"\n### {name}: task credit solved by time (240 synthetic tasks, 16 streams, 11.6 h window)")
        print("| policy | " + " | ".join(f"{h} h" for h in SLICES) + " |")
        print("|---" * (len(SLICES) + 1) + "|")
        res = {}
        for pol, p3 in (("P0 now", None), ("R: attempt 2 only without accept", None)):
            res[pol] = simulate(pol if pol.startswith("P") else "R", c1, c2, p3)
            print(f"| {pol} | " + " | ".join(f"{v:.1f}" for v in res[pol]) + " |")
        for pol, p3 in (("P0+3 (p 0.10)", 0.10), ("P0+3 (p 0.156)", 0.156), ("R+3 (p 0.10)", 0.10), ("R+3 (p 0.156)", 0.156)):
            vals = [simulate("P0" if pol.startswith("P0") else "R", c1, c2, p3, seed=s) for s in range(200)]
            m = [st.mean(v[j] for v in vals) for j in range(len(SLICES))]
            print(f"| {pol}, estimate | " + " | ".join(f"{v:.1f}" for v in m) + " |")
        tot1 = sum(min(a["tokens"], c1 or 10**9) for _, a, _ in TASKS) * SEC_PER_K / 1000 / STREAMS / 3600
        tot2 = sum(min(b["tokens"], c2 or 10**9) for _, _, b in TASKS) * SEC_PER_K / 1000 / STREAMS / 3600
        print(f"(all attempt 1s: {tot1:.1f} h of the 16 streams, all attempt 2s: {tot2:.1f} h)")


if __name__ == "__main__":
    main()
