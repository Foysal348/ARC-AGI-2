import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    colors = [c for c in np.unique(grid) if c not in (0, 1)]
    out = grid.copy()
    for c in colors:
        cells = [(r, k) for r in range(h) for k in range(w) if grid[r, k] == c]
        if not cells:
            continue
        if len(cells) == 1:
            r0, k0 = cells[0]
            # 4-connected 0-region containing the dot (dot may touch frame diagonally)
            seeds = [(a, b) for (a, b) in [(r0 - 1, k0), (r0 + 1, k0), (r0, k0 - 1), (r0, k0 + 1)]
                     if 0 <= a < h and 0 <= b < w and grid[a, b] == 0]
            reg = set()
            dq = deque(seeds)
            while dq:
                x, y = dq.popleft()
                if (x, y) in reg:
                    continue
                reg.add((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 0 and (nx, ny) not in reg:
                        dq.append((nx, ny))
            rmin, rmax = min(r for r, _ in reg), max(r for r, _ in reg)
            kmin, kmax = min(k for _, k in reg), max(k for _, k in reg)
            for r in range(rmin, rmax + 1):
                for k in range(kmin, kmax + 1):
                    if (r, k) in reg and (r, k) != (r0, k0):
                        out[r, k] = c
    return out