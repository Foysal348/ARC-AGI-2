import numpy as np
from collections import deque


def transform(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    is5 = (grid == 5)
    all5 = [(r, c) for r in range(H) for c in range(W) if is5[r, c]]

    # connected components of 5-cells (4-connectivity)
    visited = set()
    comps = []
    for start in all5:
        if start in visited:
            continue
        q = deque([start])
        visited.add(start)
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if 0 <= n[0] < H and 0 <= n[1] < W and is5[n[0], n[1]] and n not in visited:
                    visited.add(n)
                    q.append(n)
        comps.append(cells)

    # big box: component containing the top-right 5-cell
    target = None
    for cells in comps:
        if (0, W - 1) in cells:
            target = cells
            break
    if target is None:
        target = max(comps, key=len) if comps else []

    rs = [r for r, c in target]
    cs = [c for r, c in target]
    ir0, ir1 = min(rs) + 1, max(rs) - 1   # interior rows
    ic0, ic1 = min(cs) + 1, max(cs) - 1   # interior cols

    tset = set(target)
    left = [p for p in all5 if p not in tset]  # small box cells

    out = grid.copy()

    # clear the small box
    for r, c in left:
        out[r, c] = 0

    # shift the big-box interior content up by one row (clipped at top wall)
    for r in range(ir0, ir1 + 1):
        if r + 1 <= ir1:
            out[r, ic0:ic1 + 1] = grid[r + 1, ic0:ic1 + 1]
        else:
            out[r, ic0:ic1 + 1] = 0

    # redraw the small box inside the big box, horizontally centered, same rows
    lc_min = min(c for r, c in left)
    lc_max = max(c for r, c in left)
    lw = lc_max - lc_min + 1
    iw = ic1 - ic0 + 1
    dx = ic0 + (iw - lw) // 2 - lc_min
    for r, c in left:
        out[r, c + dx] = 5

    return out