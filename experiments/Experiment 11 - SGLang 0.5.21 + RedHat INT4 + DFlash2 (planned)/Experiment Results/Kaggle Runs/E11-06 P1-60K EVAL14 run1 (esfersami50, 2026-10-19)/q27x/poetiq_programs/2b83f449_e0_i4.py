import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    H, W = g.shape
    out = np.zeros((H, W), dtype=g.dtype)

    # classify rows
    wall_rows, block_rows = [], []
    for r in range(H):
        if np.any(g[r] == 7):
            block_rows.append(r)
        else:
            wall_rows.append(r)

    # 7-runs in block rows
    runs_by_row = {}
    for r in block_rows:
        runs = []
        c = 0
        while c < W:
            if g[r, c] == 7:
                s = c
                while c < W and g[r, c] == 7:
                    c += 1
                e = c - 1
                runs.append((s, (s + e) // 2, e))
            else:
                c += 1
        runs_by_row[r] = runs

    def adjacent_block(wr, dirn):
        for d in range(1, H):
            r = wr + dirn * d
            if r < 0 or r >= H:
                return None
            if r in runs_by_row:
                return r
            if g[r, r and 0] == -1:
                return None
            if r in wall_rows:
                continue
        return None

    # build base image
    for wr in wall_rows:
        out[wr] = np.where(g[wr] == 0, 0, 8)
    for br in block_rows:
        for s, m, e in runs_by_row[br]:
            out[br, s] = 8
            out[br, m] = 6
            out[br, e] = 8

    # 6s on wall rows at centers of adjacent block runs
    for wr in wall_rows:
        cs = set()
        for br in (adjacent_block(wr, -1), adjacent_block(wr, 1)):
            if br is not None:
                cs.update(m for _, m, _ in runs_by_row[br])
        for c in cs:
            if out[wr, c] == 8:
                out[wr, c] = 6

    # 3s on wall rows
    for wr in wall_rows:
        up = adjacent_block(wr, -1)
        dn = adjacent_block(wr, 1)
        Lm = Rm = False
        if up is not None and dn is not None:
            uL = min(s for s, _, _ in runs_by_row[up]); uR = max(e for _, _, e in runs_by_row[up])
            dL = min(s for s, _, _ in runs_by_row[dn]); dR = max(e for _, _, e in runs_by_row[dn])
            Lm = uL < dL
            Rm = uR > dR
        elif up is not None:
            Lm = Rm = True
        elif dn is not None:
            Lm = Rm = False

        holes = [c for c in range(W) if g[wr, c] == 0]
        if Lm:
            if g[wr, 0] == 3:
                out[wr, 0] = 3
                if not holes and out[wr, 1] == 8:
                    out[wr, 1] = 3
            else:
                ln = 0
                c = 0
                while c < W and g[wr, c] == 8:
                    ln += 1
                    c += 1
                if ln >= 3:
                    out[wr, 0] = 3
        if Rm:
            if g[wr, W - 1] == 3:
                out[wr, W - 1] = 3
                if not holes and out[wr, W - 2] == 8:
                    out[wr, W - 2] = 3
            else:
                ln = 0
                c = W - 1
                while c >= 0 and g[wr, c] == 8:
                    ln += 1
                    c -= 1
                if ln >= 3:
                    out[wr, W - 1] = 3

        # hole-adjacent markers: side with the longer 8-run
        for h in holes:
            llen = 0
            c = h - 1
            while c >= 0 and g[wr, c] == 8:
                llen += 1
                c -= 1
            rlen = 0
            c = h + 1
            while c < W and g[wr, c] == 8:
                rlen += 1
                c += 1
            if rlen > llen:
                t = h + 1
            elif llen > rlen:
                t = h - 1
            else:
                t = h + 1 if h + 1 < W else h - 1
            if 0 <= t < W and out[wr, t] == 8:
                out[wr, t] = 3

    return out