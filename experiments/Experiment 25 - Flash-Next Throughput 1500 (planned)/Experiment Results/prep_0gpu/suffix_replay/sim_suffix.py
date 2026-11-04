"""Step 2: causal replay of suffix decoding (Arctic-style) and a suffix/MTP hybrid on the E26 workload.
usage: python sim_suffix.py <phase 1|2> <ctxmode full|noreason> <out.json> [max_requests] [D list, phase 2 only]
Phase 1: suffix-only walks (D=3,8,24) + position probe (every 8th position) + calibration tables.
Phase 2: hybrid walks (raw-score / calibrated / oracle rule) at D=3,8,24 (needs <out>_cal_in.json from phase 1).
Algorithm (own reimplementation, pure Python, hash-chain index instead of a trie):
  * two trees: per-request (prompt segments + earlier turns + tokens generated so far) and global (responses completed
    before the request started; causal by wall clock t1 <= t0).
  * candidates = last 48 occurrences of the 3-gram and of the 8-gram ending at the current context end; backward match
    length l (cap 24) is measured exactly. For match length m (<= 4 distinct values tried) the occurrence group {l >= m}
    is walked greedily: next token = most frequent continuation inside the group, cumulative prob = prod(count/groupsize),
    stop when prob < 0.1; draft length <= min(D, m * spec_factor(1.0), 24 - m) (depth-24 tree). Score = sum of cumulative
    probs; best m and best tree (local/global) win.
  * step: draft d_1..d_k predicts gen[p..p+k-1]; accepted a = leading exact matches; p += a + 1 (target bonus token).
"""
import pickle, sys, json, random, time
import numpy as np
from collections import defaultdict
phase = int(sys.argv[1]); ctxmode = sys.argv[2]; OUT = sys.argv[3]
MAXREQ = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
DSEL = [int(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else [3, 8, 24]
BR = pickle.load(open("D:/ARC AGI 2/_scratch/prep_suffix/data.pkl", "rb"))
MAXD = 24; MINP = 0.1; NCAND = 48
MTP_COND = [0.864, 0.731, 0.607]
MTP_E = MTP_COND[0] + MTP_COND[0]*MTP_COND[1] + MTP_COND[0]*MTP_COND[1]*MTP_COND[2]   # expected accepted drafts (1.879)
DS = [3, 8, 24]
rng = random.Random(12345)
cal = json.load(open(OUT.replace('.json', '_cal_in.json'))) if phase == 2 else None


class Idx:
    def __init__(self):
        self.arr = []; self.h3 = {}; self.h8 = {}; self.p3 = []; self.p8 = []

    def extend(self, toks):
        for t in toks:
            self.arr.append(t); self.p3.append(-1); self.p8.append(-1)

    def ins(self, j):
        a = self.arr
        if j >= 2 and a[j-2] >= 0 and a[j-1] >= 0 and a[j] >= 0:
            k = (a[j-2] << 36) | (a[j-1] << 18) | a[j]
            self.p3[j] = self.h3.get(k, -1); self.h3[k] = j
        if j >= 7:
            w = a[j-7:j+1]
            if min(w) >= 0:
                k = hash(tuple(w)); self.p8[j] = self.h8.get(k, -1); self.h8[k] = j

    def insert_upto(self, upto, start):
        for j in range(start, upto + 1): self.ins(j)

    def cands(self, e, buf):
        a = self.arr; out = {}
        if e < 2: return []
        k3 = (buf[e-2] << 36) | (buf[e-1] << 18) | buf[e]
        chains = [(self.h3.get(k3, -1), self.p3)]
        if e >= 7:
            chains.append((self.h8.get(hash(tuple(buf[e-7:e+1])), -1), self.p8))
        for j, prev in chains:
            c = 0
            while j >= 0 and c < NCAND:
                if j not in out:
                    l = 0
                    while l < MAXD and j - l >= 0 and e - l >= 0 and a[j-l] == buf[e-l]: l += 1
                    out[j] = l
                j = prev[j]; c += 1
        return [(j, l) for j, l in out.items() if l >= 3]


def build(arr, limit, cl, m, maxlen):
    grp = [j for j, l in cl if l >= m]
    path = []; cum = []; P = 1.0; cur = grp; n = len(arr)
    for s in range(maxlen):
        cnt = {}
        for j in cur:
            q = j + 1 + s
            if q > limit or q >= n: continue
            t = arr[q]
            if t < 0: continue
            cnt.setdefault(t, []).append(j)
        if not cnt: break
        t = max(cnt, key=lambda x: len(cnt[x]))
        P *= len(cnt[t]) / len(cur)
        if P < MINP: break
        path.append(t); cum.append(P); cur = cnt[t]
    return path, cum


def best_for_tree(arr, limit, cl, D):
    if not cl: return None
    ls = sorted(set(l for _, l in cl), reverse=True)[:4]
    best = None
    for m in ls:
        maxlen = min(D, m, MAXD - m)
        if maxlen <= 0: continue
        path, cum = build(arr, limit, cl, m, maxlen)
        if path:
            sc = sum(cum)
            if best is None or sc > best[0]: best = (sc, path, cum, m)
    return best


def spec(p, buf, L, GI, memo, D):
    key = (p, D)
    if key in memo['res']: return memo['res'][key]
    e = p - 1 + memo['off']
    if p not in memo['c']:
        memo['c'][p] = (L.cands(e, buf), GI.cands(e, buf))
    cl, cg = memo['c'][p]
    bl = best_for_tree(L.arr, e, cl, D); bg = best_for_tree(GI.arr, 10**12, cg, D)
    r = None
    for src, b in (('L', bl), ('G', bg)):
        if b and (r is None or b[0] > r[0]): r = (b[0], b[1], b[2], b[3], src)
    memo['res'][key] = r
    return r


def calE(D, score):
    c = cal[str(D)]; i = int(np.searchsorted(c['edges'], score, side='right'))
    return c['mean'][min(i, len(c['mean']) - 1)]


def tbucket(k): return 0 if k == 0 else 1 if k == 1 else 2 if k < 4 else 3 if k < 8 else 4


# stats: (variant, seg, turnbucket) -> [steps, tokens, nprop, sumlen, suma_suffix, nsuffix, verify_tokens, unused]
ST = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0.0])
SRC = defaultdict(int)
PROBE = defaultdict(lambda: [0, 0, 0, 0, 0.0, 0.0, 0.0])   # (D, seg) -> n, nprop, sumlen, suma, sumscore, hybrid_raw, hybrid_oracle
CALDATA = {D: [] for D in DS}

variants = ([('S', D, None) for D in DS] if phase == 1 else
            [('H', D, rule) for D in DSEL for rule in ('raw', 'cal', 'orc')])
events = []
for bi, b in enumerate(BR):
    for k, r in enumerate(b['reqs']): events.append((r['t0'], bi, k))
events.sort(); events = events[:MAXREQ]
done = sorted((BR[bi]['reqs'][k]['t1'], bi, k) for _, bi, k in events)
GI = Idx(); di = 0; t_start = time.time(); ntok = 0
for ei, (t0, bi, k) in enumerate(events):
    while di < len(done) and done[di][0] <= t0:
        _, bj, kj = done[di]; di += 1
        ids = BR[bj]['reqs'][kj]['ids'].tolist()
        start = len(GI.arr); GI.extend(ids + [-1])
        GI.insert_upto(start + len(ids) - 2, start)
    b = BR[bi]; ctx = []
    for na, ids in b['segtok']:
        if na <= k: ctx.extend(ids.tolist())
    for kk in range(k):
        r0 = b['reqs'][kk]
        if ctxmode == 'noreason': ctx.extend(r0['ids'][r0['seg'] != 0].tolist())
        else: ctx.extend(r0['ids'].tolist())
    r = b['reqs'][k]; gen = r['ids'].tolist(); seg = r['seg']; n = len(gen)
    buf = ctx + gen; off = len(ctx)
    L = Idx(); L.arr = buf; L.p3 = [-1] * len(buf); L.p8 = [-1] * len(buf)
    L.insert_upto(off - 2, 0)
    ins_to = [off - 2]
    memo = dict(res={}, c={}, off=off)
    tb = tbucket(k)

    def ensure(p):
        e = p - 1 + off
        while ins_to[0] < e - 1:
            ins_to[0] += 1; L.ins(ins_to[0])

    for var in variants:
        kind, D, rule = var; vname = '%s%d%s' % (kind, D, ('_' + rule) if rule else '')
        p = 0
        # the local index must only know tokens before p; walks are monotone but run one after another, so rebuild
        # the local index per variant (cheap: insertion is incremental inside this variant)
        L = Idx(); L.arr = buf; L.p3 = [-1] * len(buf); L.p8 = [-1] * len(buf)
        L.insert_upto(off - 2, 0); ins_to[0] = off - 2
        memo = dict(res={}, c={}, off=off)
        while p < n:
            ensure(p)
            sp = spec(p, buf, L, GI, memo, D)
            s = int(seg[p]); a_s = 0; ln = 0; use_suffix = False
            if sp:
                path = sp[1]; ln = len(path)
                while a_s < ln and p + a_s < n and path[a_s] == gen[p + a_s]: a_s += 1
            if kind == 'S':
                use_suffix = ln > 0
            elif ln > 0:
                if rule == 'raw': use_suffix = sp[0] > MTP_E
                elif rule == 'cal': use_suffix = calE(D, sp[0]) > MTP_E
                else: use_suffix = a_s > MTP_E
            if use_suffix:
                a = a_s; ver = ln + 1; src = sp[4]
            elif kind == 'S':
                a = 0; ver = 1; src = '-'
            else:
                a = 0
                for c in MTP_COND:
                    if rng.random() < c: a += 1
                    else: break
                a = max(0, min(a, n - p - 1)); ver = 4; src = 'M'
            if kind == 'S' and sp:
                CALDATA[D].append((sp[0], a_s, ln))
            st = ST[(vname, s, tb)]
            st[0] += 1; st[1] += min(a + 1, n - p); st[6] += ver
            if ln > 0: st[2] += 1; st[3] += ln
            if use_suffix: st[5] += 1; st[4] += a
            SRC[(vname, src)] += 1
            p += a + 1
    if phase == 1:
        L = Idx(); L.arr = buf; L.p3 = [-1] * len(buf); L.p8 = [-1] * len(buf)
        L.insert_upto(off - 2, 0); ins_to[0] = off - 2
        memo = dict(res={}, c={}, off=off)
        for p in range(0, n, 8):
            ensure(p); s = int(seg[p])
            for D in DS:
                sp = spec(p, buf, L, GI, memo, D)
                pr = PROBE[(D, s)]; pr[0] += 1
                if sp:
                    path = sp[1]; a_s = 0
                    while a_s < len(path) and p + a_s < n and path[a_s] == gen[p + a_s]: a_s += 1
                    pr[1] += 1; pr[2] += len(path); pr[3] += a_s; pr[4] += sp[0]
                    pr[5] += a_s if sp[0] > MTP_E else MTP_E
                    pr[6] += max(a_s, MTP_E)
                else:
                    pr[5] += MTP_E; pr[6] += MTP_E
    ntok += n
    if ei % 25 == 0:
        print('req', ei, '/', len(events), 'tok', ntok, 'sec %.0f' % (time.time() - t_start), flush=True)

res = dict(phase=phase, ctxmode=ctxmode, MTP_E=MTP_E, ntok=ntok,
           ST={'|'.join(map(str, k)): v for k, v in ST.items()},
           SRC={'|'.join(map(str, k)): v for k, v in SRC.items()},
           PROBE={'|'.join(map(str, k)): v for k, v in PROBE.items()})
if phase == 1:
    calout = {}
    for D in DS:
        arr = np.array(CALDATA[D])
        if len(arr) < 100: continue
        q = np.unique(np.quantile(arr[:, 0], np.linspace(0, 1, 13)[1:-1]))
        idx = np.searchsorted(q, arr[:, 0], side='right')
        mean = [float(arr[idx == i, 1].mean()) if (idx == i).any() else 0.0 for i in range(len(q) + 1)]
        calout[str(D)] = dict(edges=q.tolist(), mean=mean, n=[int((idx == i).sum()) for i in range(len(q) + 1)])
    res['cal'] = calout
    json.dump(calout, open(OUT.replace('.json', '_cal.json'), 'w'))
json.dump(res, open(OUT, 'w'))
print('DONE', time.time() - t_start)
