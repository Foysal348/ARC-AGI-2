import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    H, W = g.shape

    def is_wall(r):
        row = g[r]
        cnt = {}
        for v in row:
            cnt[int(v)] = cnt.get(int(v), 0) + 1
        return cnt.get(8, 0) >= max(1, (W // 2) + 1)

    wall_rows = [r for r in range(H) if is_wall(r)]
    if not wall_rows:
        return g.copy()

    first, last = wall_rows[0], wall_rows[-1]
    block_rows = [r for r in range(H) if r < last and r not in wall_rows]

    centers = {}
    for r in block_rows:
        cs = []
        c = 0
        while c < W:
            if g[r, c] == 7:
                s = c
                while c < W and g[r, c] == 7:
                    c += 1
                e = c - 1
                cs.append((s, (s + e) // 2))
            else:
                c += 1
        centers[r] = cs

    def block_rows_adjacent(wr):
        up = [r for r in block_rows if r < wr and all(r2 > r for r2 in wall_rows if r2 < wr)]
        dn = [r for r in block_rows if r > wr and all(r2 < r for r2 in wall_rows if r2 > wr)]
        return (up[-1] if up else None, dn[0] if dn else None)

    def gap(bs, side):
        d = None
        for s, e in bs:
            dd = s if side == 'L' else (W - 1 - e)
            d = dd if d is None else min(d, dd)
        return d if d is not None else 10 ** 9

    def hole3s(wr):
        cells = []
        hole_cols = [c for c in range(W) if g[wr, c] == 0]
        mid = (W - 1) / 2.0
        for c in hole_cols:
            if c == 0:
                cand = 1
            elif c == W - 1:
                cand = W - 2
            else:
                cand = c + 1 if c < mid else c - 1
            if 0 <= cand < W and g[wr, cand] == 8:
                cells.append(cand)
        return cells

    out = g.copy()
    for r in block_rows:
        for s, e in centers[r]:
            out[r, s] = 8
            out[r, e] = 8
            out[r, (s + e) // 2] = 6

    for wr in wall_rows:
        for c in range(W):
            if g[wr, c] == 8:
                out[wr, c] = 8
        a, b = block_rows_adjacent(wr)
        cs = set()
        if a is not None:
            cs.update(x[1] for x in centers[a])
        if b is not None:
            cs.update(x[1] for x in centers[b])
        for c in cs:
            if g[wr, c] == 8:
                out[wr, c] = 6
        if wr == first:
            continue

        marked = []
        if wr == last:
            if a is not None:
                dL = gap(centers[a], 'L')
                dR = gap(centers[a], 'R')
                if dL <= 3 or dL >= 6:
                    marked.append(0)
                if dR <= 3 or dR >= 6:
                    marked.append(W - 1)
        else:
            if a is not None:
                if gap(centers[a], 'L') <= 3 and gap(centers[b], 'L') > 3:
                    marked += [0, 1]
                if gap(centers[a], 'R') <= 3 and gap(centers[b], 'R') > 3:
                    marked += [W - 2, W - 1]

        for c in marked:
            if out[wr, c] == 8:
                out[wr, c] = 3
        for c in hole3s(wr):
            out[wr, c] = 3

    return out