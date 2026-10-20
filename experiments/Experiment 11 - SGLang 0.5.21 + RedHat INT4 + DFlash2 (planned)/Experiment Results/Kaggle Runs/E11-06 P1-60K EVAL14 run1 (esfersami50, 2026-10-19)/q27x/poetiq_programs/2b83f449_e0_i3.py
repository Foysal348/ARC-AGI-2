import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()

    wall_rows = []
    block_rows = []
    for r in range(H):
        if np.any(g[r] == 7):
            block_rows.append(r)
        else:
            wall_rows.append(r)

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

    for r in wall_rows:
        up = r - 1
        if up in runs_by_row:
            for s, m, e in runs_by_row[up]:
                if out[r, m] == 8:
                    out[r, m] = 6
                if s > 0 and out[r, s - 1] == 8:
                    out[r, s - 1] = 6
                if e < W - 1 and out[r, e + 1] == 8:
                    out[r, e + 1] = 6

    for r in block_rows:
        out[r, :] = 0
        for s, m, e in runs_by_row[r]:
            out[r, s] = 8
            out[r, m] = 6
            out[r, e] = 8

    if not wall_rows:
        return out

    first_wall = wall_rows[0]

    for idx, r in enumerate(wall_rows):
        if idx == 0:
            continue

        above = r - 1
        if above not in runs_by_row:
            continue

        rs = runs_by_row[above]
        L = min(s for s, _, _ in rs)
        R = max(e for _, _, e in rs)

        if L <= 3:
            for c in (0, 1):
                if 0 <= c < W and out[r, c] == 8:
                    out[r, c] = 3
        elif L == 4:
            if out[r, 0] == 8:
                out[r, 0] = 3

        dR = W - 1 - R
        if dR <= 3:
            for c in (W - 2, W - 1):
                if 0 <= c < W and out[r, c] == 8:
                    out[r, c] = 3

        holes = [c for c in range(W) if g[r, c] == 0]
        for h in holes:
            left_len = 0
            c = h - 1
            while c >= 0 and g[r, c] != 0:
                left_len += 1
                c -= 1

            right_len = 0
            c = h + 1
            while c < W and g[r, c] != 0:
                right_len += 1
                c += 1

            if right_len > left_len:
                target = h + 1
            elif left_len > right_len:
                target = h - 1
            else:
                target = h + 1 if h + 1 < W else h - 1

            if 0 <= target < W and out[r, target] == 8:
                out[r, target] = 3

    return out