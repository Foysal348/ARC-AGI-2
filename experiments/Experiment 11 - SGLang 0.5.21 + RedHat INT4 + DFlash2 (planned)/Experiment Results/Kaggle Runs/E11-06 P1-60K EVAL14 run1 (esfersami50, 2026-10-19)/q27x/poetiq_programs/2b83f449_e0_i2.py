import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    H, W = g.shape
    out = np.zeros_like(g)

    # classify rows: wall rows (mostly 8) vs block rows (contain 7s)
    wall_rows = []
    block_rows = []
    for r in range(H):
        has7 = np.any(g[r] == 7)
        n8 = int(np.sum(g[r] == 8))
        if has7 and n8 == 0:
            block_rows.append(r)
        else:
            wall_rows.append(r)

    # block runs in each block row -> centers
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
                cs.append((s, (s + e) // 2, e))
            else:
                c += 1
        centers[r] = cs

    def neighbor_block_row(wr, dirn):
        cand = None
        for r in block_rows:
            if (dirn == -1 and r < wr) or (dirn == 1 and r > wr):
                if cand is None or abs(r - wr) < abs(cand - wr):
                    cand = r
                else:
                    break
        # ensure adjacency (no other wall row between)
        if cand is None:
            return None
        if dirn == -1:
            if any(w for w in wall_rows if cand < w < wr):
                return None
        else:
            if any(w for w in wall_rows if wr < w < cand):
                return None
        return cand

    first, last = wall_rows[0], wall_rows[-1]

    for wr in wall_rows:
        # base: 8s with holes kept; block-row patterns applied below
        out[wr] = np.where(g[wr] == 0, 0, 8)

        # 6s at centers of adjacent block rows
        up = neighbor_block_row(wr, -1)
        dn = neighbor_block_row(wr, 1)
        cs = set()
        if up is not None:
            cs.update(x[1] for x in centers[up])
        if dn is not None:
            cs.update(x[1] for x in centers[dn])
        for c in cs:
            if g[wr, c] == 8:
                out[wr, c] = 6

        # edge markers from nearest-block distance to the edge
        a = up if up is not None else dn
        if a is not None:
            dL = min(x[0] for x in centers[a])
            dR = W - 1 - max(x[2] for x in centers[a])
            if dL <= 3:
                for c in (0, 1):
                    out[wr, c] = 3
            if dR <= 3:
                for c in (W - 2, W - 1):
                    out[wr, c] = 3

        # hole-adjacent markers
        for c in range(W):
            if g[wr, c] == 0:
                if c > 0 and g[wr, c - 1] == 8:
                    out[wr, c - 1] = 3
                if c < W - 1 and g[wr, c + 1] == 8:
                    out[wr, c + 1] = 3

    # block rows: zero background, 8 6 8 at each former 7-run
    for r in block_rows:
        out[r] = 0
        for s, m, e in centers[r]:
            out[r, s] = 8
            out[r, e] = 8
            out[r, m] = 6

    return out