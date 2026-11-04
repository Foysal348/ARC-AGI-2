"""E33 attempt-chain coverage sim on the real E28 K100 EVAL120 attempt data (constant per-stream speed, 24 slots)."""
import json, heapq, sys, random
base = r"D:/ARC AGI 2/_scratch/e28_k100_out/q27-xhigh-e28-k100-eval120-6h/q27x/e28_k100_report.json"
d = json.load(open(base, encoding="utf-8"))
sa = d["sched_attempts"]
A = sorted([a for a in sa if a["phase"] == "A"], key=lambda a: a["start_seq"])
B2 = {a["task"]: a for a in sa if a["phase"] == "B2"}
C = {a["task"]: a for a in sa if a["phase"] == "C"}
HIT2 = {"31f7f899", "6e453dd6", "78332cb0", "7ed72f31", "db695cfb", "dd6b8c4b"}


def run(window_min, rate, copies=1, slots=24, p_poe=0.15, p_a2hard=0.5, seed=0, chain=True, e28_mode=False, depth_first=False, g2_first=False):
    rnd = random.Random(seed)
    tasks = []
    for c in range(copies):
        for a in A:
            tid = a["task"] + ("" if c == 0 else f"#{c}")
            tasks.append(dict(tid=tid, base=a["task"], a_tok=a["tokens"], a_pass=a["kind"] == "TRAIN_PASS",
                              b=B2.get(a["task"]), c=C.get(a["task"])))
    W = window_min * 60
    pend = []          # (prio, seq, item)
    seq = [0]
    def push(prio, item):
        seq[0] += 1; heapq.heappush(pend, (prio, seq[0], item))
    for i, t in enumerate(tasks):
        push((1, 0, 0, i), ("A", 1, t))
    running = []       # (end, item)
    now = 0.0
    stats = {}
    hits2 = set(); a2_done = set()
    def cnt(k): stats[k] = stats.get(k, 0) + 1
    def need_sec(stage):
        return {"C": 65.6 * 60, "D": 65.6 * 60, "E": 82.6 * 60}.get(stage, 0)   # notebook gates: 60k floor at a fixed 18 tok/s + drain (Poetiq 9 min, Direct 26 min) + 60 s
    while True:
        # admit
        while len(running) < slots and pend:
            prio, _, (stage, att, t) = heapq.heappop(pend)
            if stage in "CDE" and W - now < need_sec(stage):
                cnt(f"{att}{stage}_skip_time"); continue
            if stage == "A":
                tok = t["a_tok"] if att == 1 else (104000 if not t["a_pass"] else (t["c"]["tokens"] if t["c"] and t["c"]["kind"] == "TRAIN_PASS" else t["a_tok"]))
                ok = t["a_pass"] if att == 1 else (rnd.random() < p_a2hard if not t["a_pass"] else True)
            elif stage == "B":
                tok = t["b"]["tokens"] if att == 1 else 110000; ok = (t["b"]["kind"] == "TRAIN_PASS") if att == 1 else rnd.random() < 0.3
            elif stage in "CD":
                tok = 64000; ok = rnd.random() < p_poe
            else:
                tok = 104000; ok = False
            dur = tok / rate
            cnt(f"{att}{stage}_start")
            heapq.heappush(running, (now + dur, seq[0], (stage, att, t, ok))); seq[0] += 1
        if not running:
            break
        end, _, (stage, att, t, ok) = heapq.heappop(running)
        if end > W:
            cnt(f"{att}{stage}_cut"); now = max(now, W) if False else now
            # everything still running is cut at the window
            for e2, _, (s2, a2, t2, _) in running:
                cnt(f"{a2}{s2}_cut")
            break
        now = end
        cnt(f"{att}{stage}_done")
        if att == 2 and stage == "A": cnt("2A_g%d_done" % (0 if not t["a_pass"] else 1))
        if att == 1 and stage == "A" and not any(x[2][0] == "A" and x[2][1] == 1 for x in pend): stats.setdefault("A1_queue_empty_min", round(now/60))
        if ok:
            cnt(f"{att}{stage}_pass")
            if att == 2 and t["base"] in HIT2 and t["tid"] == t["base"]:
                hits2.add(t["base"])
        if att == 1 and stage == "A" and not e28_mode:
            g = (0 if not t["a_pass"] else 1) ^ (1 if g2_first else 0)
            key = (tasks.index(t),) if g == 0 else (t["a_tok"], tasks.index(t))
            push((2, 0, g) + key, ("A", 2, t))
        if e28_mode and att == 1 and stage == "A" and t["a_pass"]:
            push((3, t["a_tok"], 0, 0), ("A", 2, t))
        if not ok and chain:
            nxt = {"A": "B" if (att == 1 and t["b"]) or att == 2 else "C", "B": "C", "C": "D", "D": "E"}.get(stage)
            if e28_mode and stage != "A":
                nxt = None
            if nxt:
                si = "ABCDE".index(nxt)
                push((att, (-si if depth_first else si), 0, tasks.index(t)), (nxt, att, t))
    return stats, sorted(hits2)


if __name__ == "__main__":
    keys = ("A1_queue_empty_min", "1B_done", "1B_pass", "1C_start", "1C_done", "1D_start", "1D_done", "1E_start", "1E_done", "1C_skip_time", "1D_skip_time", "1E_skip_time", "2A_start", "2A_done", "2A_g0_done", "2A_g1_done", "2A_cut", "2B_start", "2C_start")
    for label, win, copies in (("EVAL120 6h", 360, 1), ("hidden 12h (240)", 670, 2), ("hidden 12h (240) w690", 690, 2)):
        for f in (1.0, 1.04, 1.1, 1.16):
            for mode in ("E33", "E28"):
                s, h = run(win, 17.9 * f, copies, e28_mode=(mode == "E28"))
                print(f"{label:22s} x{f:<5} {mode:4s}", " ".join(f"{k}={s.get(k,0)}" for k in keys), "| a2hit6:", len(h))
