import json, glob, os, sys, collections
G = 64
root = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(root, "duck", "*", "branch_result.json"))):
    b = json.load(open(f))
    u = b.get("usage") or []
    br = os.path.basename(os.path.dirname(f))
    for k in range(1, len(u)):
        p0, c0, p1, ca = u[k-1]["prompt"], u[k-1]["completion"], u[k]["prompt"], u[k]["cached"]
        if ca is None:
            continue
        A = p0 + c0
        full = (A // G) * G
        pre = (p0 // G) * G
        if p1 < p0:
            cls = "history_shrank"           # trim / reset / new run: prefix changed by design
        elif ca >= full - G:
            cls = "reply_reused"
        elif abs(ca - pre) <= G:
            cls = "reply_lost_at_prefill_end"
        elif ca < pre - G:
            cls = "deeper_miss"
        else:
            cls = "partial_reply"
        lost = max(0, min(A, p1) - ca)   # tokens of the shared-able prefix that were prefilled again
        rows.append(dict(br=br, k=k, p0=p0, c0=c0, p1=p1, cached=ca, cls=cls, lost=lost, reply=c0))
cnt = collections.Counter(r["cls"] for r in rows)
lost = collections.Counter()
for r in rows:
    lost[r["cls"]] += r["lost"]
print("requests with a predecessor:", len(rows))
for c, n in cnt.most_common():
    print(f"  {c:28s} {n:4d} ({100*n/len(rows):.0f} %)  re-prefilled prefix tokens {lost[c]:7d}")
print("total re-prefilled prefix tokens", sum(lost.values()), " total prompt tokens", sum(r["p1"] for r in rows),
      " total uncached", sum(r["p1"] - r["cached"] for r in rows))
# where inside the reply did the lost cases stop? relative to p0
for c in ("reply_lost_at_prefill_end", "partial_reply"):
    d = [r["cached"] - r["p0"] for r in rows if r["cls"] == c]
    if d:
        d.sort(); print(c, "cached - prompt_prev: min/median/max", d[0], d[len(d)//2], d[-1])
rl = [r for r in rows if r["cls"] == "reply_lost_at_prefill_end"]
rr = [r for r in rows if r["cls"] == "reply_reused"]
import statistics as S
if rl and rr:
    print("reply length (completion) median: lost", S.median(r["c0"] for r in rl), " reused", S.median(r["c0"] for r in rr))
json.dump(rows, open("rows.json", "w"), indent=0)
