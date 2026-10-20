import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    H, W = g.shape

    # 1) Find the vertical wall: a full column of one non-zero color
    wall_col = None
    wall_color = None
    for c in range(W):
        col = g[:, c]
        v0 = int(col[0])
        if v0 != 0 and (col == v0).all():
            wall_col = c
            wall_color = int(v0)
            break
    if wall_col is None:
        return g.copy()

    # 2) Border color of the target room (right of wall) = dominant non-zero, non-wall color
    right = g[:, wall_col + 1:]
    vals, counts = np.unique(right, return_counts=True)
    border_color = None
    best = -1
    for v, cnt in zip(vals, counts):
        if v != 0 and int(v) != wall_color and int(cnt) > best:
            best = int(cnt)
            border_color = int(v)
    if border_color is None:
        return g.copy()

    # 3) Bounding box of the border ring
    bc_mask = (right == border_color)
    rr = np.where(bc_mask.any(axis=1))[0]
    cc = np.where(bc_mask.any(axis=0))[0]
    top, bottom = int(rr.min()), int(rr.max())
    left = wall_col + 1 + int(cc.min())
    right_edge = wall_col + 1 + int(cc.max())

    # 4) Read markers on the ring boundary: per color, rows (left/right side) and cols (top/bottom side)
    mrows = {}
    mcols = {}

    def add_marker(r, c, v):
        if v == 0 or v == border_color or v == wall_color:
            return
        if r == top or r == bottom:
            mcols.setdefault(v, set()).add(c)
        if c == left or c == right_edge:
            mrows.setdefault(v, set()).add(r)

    for c in range(left, right_edge + 1):
        add_marker(top, c, int(g[top, c]))
        add_marker(bottom, c, int(g[bottom, c]))
    for r in range(top, bottom + 1):
        add_marker(r, left, int(g[r, left]))
        add_marker(r, right_edge, int(g[r, right_edge]))

    # 5) Objects in the source area (left of the wall), grouped by color
    objects = {}
    for r in range(H):
        for c in range(wall_col):
            v = int(g[r, c])
            if v != 0:
                objects.setdefault(v, []).append((r, c))

    out = g.copy()

    # 6) Move objects to every marker crosshair (row x col), keeping shape (center-anchored).
    #    Higher colors are stamped first so lower colors overwrite on overlap.
    for color in sorted(objects.keys(), reverse=True):
        cells = objects[color]
        rs = mrows.get(color)
        cs = mcols.get(color)
        if not rs or not cs:
            continue  # no crosshair for this color: object stays put
        for (r, c) in cells:
            out[r, c] = 0
        rmin = min(p[0] for p in cells)
        rmax = max(p[0] for p in cells)
        cmin = min(p[1] for p in cells)
        cmax = max(p[1] for p in cells)
        cr = (rmin + rmax) // 2
        ccx = (cmin + cmax) // 2
        mask = [(r - cr, c - ccx) for (r, c) in cells]
        for tr in sorted(rs):
            for tc in sorted(cs):
                for dr, dc in mask:
                    rr2 = tr + dr
                    cc2 = tc + dc
                    if 0 <= rr2 < H and 0 <= cc2 < W:
                        out[rr2, cc2] = color

    return out