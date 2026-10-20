import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = grid.copy()

    # ---- collect 0-cells and non-bg/non-0 cells ----
    is_zero = (grid == 0)
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # 4-connected components of 0-cells
    visited = np.zeros((h, w), dtype=bool)
    zero_comps = []
    for r in range(h):
        for c in range(w):
            if is_zero[r, c] and not visited[r, c]:
                comp = []
                dq = deque([(r, c)])
                visited[r, c] = True
                while dq:
                    rr, cc = dq.popleft()
                    comp.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and is_zero[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            dq.append((nr, nc))
                zero_comps.append(comp)

    # corners of each 0-component: cells with >=2 zero-neighbors
    def corners_of(comp):
        s = set(comp)
        res = []
        for (rr, cc) in comp:
            n = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (rr + dr, cc + dc) in s:
                    n += 1
            if n >= 2:
                res.append((rr, cc))
        return res

    comp_corners = [corners_of(comp) for comp in zero_comps]
    all_corners = [p for cc in comp_corners for p in cc]

    # dots
    dots = []
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            if v != bg and v != 0:
                dots.append((r, c, v))

    # ---- pre-existing structures (color cells adjacent to 0s) -> anchor cells ----
    anchored = set()
    for (r, c, v) in dots:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and is_zero[nr, nc] and (nr, nc) not in anchored:
                    anchored.add((nr, nc))

    def nearest_corner(r, c):
        best, bestd = None, None
        for (cr, ck) in all_corners:
            d = abs(cr - r) + abs(ck - c)
            if bestd is None or d < bestd:
                bestd, best = d, (cr, ck)
        return best

    def stamp(corner, dot, color, skip_0):
        cr, ck = corner
        dr = 0 if dot[0] == cr else (1 if dot[0] > cr else -1)
        dc = 0 if dot[1] == ck else (1 if dot[1] > ck else -1)
        cells = []
        if dr != 0 and dc != 0:  # diagonal dot: plus + axis diagonals
            cells = [(cr + dr, ck), (cr, ck + dc), (cr - dr, ck), (cr, ck - dc),
                     (cr + dr, ck + dc), (cr - dr, ck - dc)]
        elif dr != 0:  # vertical dot: N, S, W, E
            cells = [(cr - dr, ck), (cr + dr, ck), (cr, ck - 1), (cr, ck + 1)]
        else:  # horizontal dot: N, S, W, E
            cells = [(cr - 1, ck), (cr + 1, ck), (cr, ck - dc), (cr, ck + dc)]
        for (rr, cc) in cells:
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if skip_0 and is_zero[rr, cc] and (rr, cc) != corner:
                continue
            out[rr, cc] = color

    # ---- process each dot with its corner ----
    for (r, c, v) in dots:
        rmin = min(p[0] for p in [(r, c)])
        cmin = min(p[1] for p in [(r, c)])
        rmax = max(p[0] for p in [(r, c)])
        cmax = max(p[1] for p in [(r, c)])
        containing = [i for i, comp in enumerate(zero_comps)
                      if rmin >= min(p[0] for p in comp) and rmax <= max(p[0] for p in comp)
                      and cmin >= min(p[1] for p in comp) and cmax <= max(p[1] for p in comp)]
        if containing:
            best, bestd = None, None
            for i in containing:
                for (cr, ck) in comp_corners[i]:
                    d = abs(cr - r) + abs(ck - c)
                    if bestd is None or d < bestd:
                        bestd, best = d, (cr, ck)
            corner = best
        else:
            # nearest corner overall, but avoid cells of an already-anchored structure
            cands = [p for p in all_corners if p not in anchored]
            if not cands:
                cands = all_corners
            best, bestd = None, None
            for (cr, ck) in cands:
                d = abs(cr - r) + abs(ck - c)
                if bestd is None or d < bestd:
                    bestd, best = d, (cr, ck)
            corner = best
        if corner is None:
            continue
        stamp(corner, (r, c), v, skip_0=(r - corner[0]) * (c - corner[1]) > 0)

    return out