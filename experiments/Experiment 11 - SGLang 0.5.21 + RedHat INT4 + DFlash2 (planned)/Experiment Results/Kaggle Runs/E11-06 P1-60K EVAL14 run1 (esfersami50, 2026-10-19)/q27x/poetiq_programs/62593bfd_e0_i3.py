import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    if h == 0 or w == 0:
        return grid.copy()

    bg = int(np.bincount(grid.ravel()).argmax())

    def neighbors(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
        if seen[r, c] or grid[r, c] == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in neighbors(cr, cc):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
            components.append((color, cells))

    out = np.full((h, w), bg, dtype=np.int64)
    for color, cells in components:
        rs = [r for r, _ in cells]
        r0 = min(rs)
        r1 = max(rs)
        # symmetric objects go to the top edge, others to the bottom edge
        if r0 <= r1:
            shift = -r0
        else:
            shift = (h - 1) - r1
        for r, c in cells:
            nr = r + shift
            if 0 <= nr < h:
                out[nr, c] = color
    return out