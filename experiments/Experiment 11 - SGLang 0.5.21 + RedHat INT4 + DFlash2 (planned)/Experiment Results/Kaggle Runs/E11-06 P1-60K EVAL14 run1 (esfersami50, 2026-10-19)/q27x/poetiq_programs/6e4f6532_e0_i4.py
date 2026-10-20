import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    h, w = g.shape
    out = g.copy()
    if h < 2 or w < 2:
        return out

    # --- detect 2-px uniform bars ---
    row_bars = []
    for r in range(h - 1):
        if len(np.unique(g[r])) == 1 and len(np.unique(g[r + 1])) == 1 and g[r, 0] == g[r + 1, 0]:
            row_bars.append([r, r + 1, int(g[r, 0])])

    rbr = set()
    for rb in row_bars:
        rbr.add(rb[0]); rbr.add(rb[1])

    col_bars = []
    for c in range(w - 1):
        idx = [r for r in range(h) if r not in rbr]
        if idx:
            a, b = g[idx, c], g[idx, c + 1]
            if len(np.unique(a)) == 1 and len(np.unique(b)) == 1 and a[0] == b[0]:
                col_bars.append([c, c + 1, int(a[0])])

    frame = np.zeros((h, w), dtype=bool)
    for rb in row_bars:
        frame[rb[0]:rb[1] + 1, :] = True
    for cb in col_bars:
        frame[:, cb[0]:cb[1] + 1] = True

    nf = g[~frame]
    bg = int(np.unique(nf, return_counts=True)[0][np.argmax(np.unique(nf, return_counts=True)[1])]) if nf.size else 0
    fcols = set(rb[2] for rb in row_bars) | set(cb[2] for cb in col_bars)

    # --- regions (connected non-frame areas) ---
    rid = np.full((h, w), -1)
    regions = []
    for i in range(h):
        for j in range(w):
            if not frame[i, j] and rid[i, j] == -1:
                q, cells = deque([(i, j)]), []
                rid[i, j] = len(regions)
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not frame[ny, nx] and rid[ny, nx] == -1:
                            rid[ny, nx] = len(regions); q.append((ny, nx))
                regions.append(cells)

    rsc = {}
    for ri, rc in enumerate(regions):
        ys = [y for y, _ in rc]; xs = [x for _, x in rc]
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        sc = {'U': set(), 'D': set(), 'L': set(), 'R': set()}
        for rb in row_bars:
            if rb[1] == y0 - 1: sc['U'].add(rb[2])
            if rb[0] == y1 + 1: sc['D'].add(rb[2])
        for cb in col_bars:
            if cb[1] == x0 - 1: sc['L'].add(cb[2])
            if cb[0] == x1 + 1: sc['R'].add(cb[2])
        rsc[ri] = sc

    # --- components ---
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for i in range(h):
        for j in range(w):
            if not vis[i, j] and g[i, j] != bg and not frame[i, j]:
                q, cells = deque([(i, j)]), []
                vis[i, j] = True
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not vis[ny, nx] and g[ny, nx] != bg and not frame[ny, nx]:
                            vis[ny, nx] = True; q.append((ny, nx))
                comps.append(cells)

    e8 = [c for c in comps if any(g[y, x] == 8 for y, x in c)]
    if not e8:
        return out

    for c in e8:
        for y, x in c:
            out[y, x] = bg

    def dihs(m):
        yield m
        yield np.rot90(m, 1)
        yield np.rot90(m, 2)
        yield np.rot90(m, 3)
        yield np.fliplr(m)
        yield np.flipud(m)
        yield np.rot90(np.fliplr(m), 1)
        yield np.rot90(np.fliplr(m), 2)

    for c in e8:
        ys = [y for y, _ in c]; xs = [x for _, x in c]
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        m = np.full((y1 - y0 + 1, x1 - x0 + 1), bg, dtype=g.dtype)
        for y, x in c:
            m[y - y0, x - x0] = g[y, x]

        ri = int(rid[y0, x0])
        sc = rsc.get(ri, {'U': set(), 'D': set(), 'L': set(), 'R': set()})

        def sc2(m2, sc=sc):
            s, hh, ww = 0, m2.shape[0], m2.shape[1]
            for cc in range(ww):
                v = int(m2[0, cc])
                if v != bg and v in fcols: s += 2 if v in sc['U'] else -1
                v = int(m2[-1, cc])
                if v != bg and v in fcols: s += 2 if v in sc['D'] else -1
            for rr in range(hh):
                v = int(m2[rr, 0])
                if v != bg and v in fcols: s += 2 if v in sc['L'] else -1
                v = int(m2[rr, ww - 1])
                if v != bg and v in fcols: s += 2 if v in sc['R'] else -1
            return s

        bm, bs = m, -9999
        for tm in dihs(m):
            s = sc2(tm)
            if s > bs:
                bs, bm = s, tm

        th, tw = bm.shape
        rc = regions[ri]
        rys = [y for y, _ in rc]; rxs = [x for _, x in rc]
        ry0, ry1, rx0, rx1 = min(rys), max(rys), min(rxs), max(rxs)
        ny = max(ry0, min((ry0 + ry1 - th) // 2, ry1 - th + 1))
        nx = max(rx0, min((rx0 + rx1 - tw) // 2, rx1 - tw + 1))

        for yy in range(th):
            for xx in range(tw):
                v = int(bm[yy, xx])
                if v == bg:
                    continue
                ay, ax = ny + yy, nx + xx
                if not (0 <= ay < h and 0 <= ax < w) or frame[ay, ax]:
                    continue
                # recolor border tags to match region side color
                if v in fcols:
                    sides = []
                    if yy == 0: sides.append('U')
                    if yy == th - 1: sides.append('D')
                    if xx == 0: sides.append('L')
                    if xx == tw - 1: sides.append('R')
                    nv = v
                    for s in sides:
                        if v in sc.get(s, set()):
                            nv = v; break
                    else:
                        for s in sides:
                            if sc.get(s):
                                nv = list(sc[s])[0]; break
                    out[ay, ax] = nv
                else:
                    out[ay, ax] = v

    return out