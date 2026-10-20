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

    def is_symmetric(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1

        s = {((r - r0), (c - c0)) for r, c in cells}

        if hh <= 1 or ww <= 1:
            return True

        # horizontal reflection
        if all((hh - 1 - r, c) in s for r, c in s):
            return True

        # vertical reflection
        if all((r, ww - 1 - c) in s for r, c in s):
            return True

        if hh == ww:
            # main diagonal reflection
            if all((c, r) in s for r, c in s):
                return True

            # anti-diagonal reflection
            if all((ww - 1 - c, hh - 1 - r) in s for r, c in s):
                return True

        return False

    out = np.full((h, w), bg, dtype=grid.dtype)

    for color, cells in components:
        sym = is_symmetric(cells)
        if sym:
            r0 = min(r for r, _ in cells)
            shift = -r0
        else:
            r1 = max(r for r, _ in cells)
            shift = (h - 1) - r1

        for r, c in cells:
            nr = r + shift
            if 0 <= nr < h:
                out[nr, c] = color

    return out