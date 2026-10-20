import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    out = grid.copy()

    # 4-connected components of 0 cells
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 and not visited[r, c]:
                comp = set()
                dq = deque([(r, c)])
                visited[r, c] = True
                while dq:
                    rr, cc = dq.popleft()
                    comp.add((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            dq.append((nr, nc))
                comps.append(comp)

    def corners(comp):
        s = comp
        res = []
        for (rr, cc) in s:
            n = sum(1 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (rr + dr, cc + dc) in s)
            if n >= 2:
                res.append((rr, cc))
        return res

    all_corners = [p for comp in comps for p in corners(comp)]

    # single-cell dots (non-bg, non-0, not 8-adjacent to same color)
    nonbg = {}
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            if v != bg and v != 0:
                nonbg.setdefault(v, []).append((r, c))

    dots = []
    for v, cells in nonbg.items():
        if len(cells) == 1:
            dots.append((cells[0][0], cells[0][1], v))
        else:
            s = set(cells)
            for (r, c) in cells:
                if any((r + dr, c + dc) in s for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                       if (dr, dc) != (0, 0)):
                    dots.append((r, c, v))

    for (r, c, v) in dots:
        best, bestd = None, None
        for (cr, ck) in all_corners:
            d = abs(cr - r) + abs(ck - c)
            if bestd is None or d < bestd:
                bestd, best = d, (cr, ck)
        if best is None:
            continue
        cr, ck = best
        ar = (r > cr) - (r < cr)
        ac = (c > ck) - (c < ck)
        if ar == 0 or ac == 0:
            cells = [(cr + ar, ck + ac)]
        else:
            cells = [(cr + ar, ck + ac),
                     (cr - ar, ck - ac),
                     (cr - ar, ck),
                     (cr, ck - ac)]
        for (rr, cc) in cells:
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = v
    return out