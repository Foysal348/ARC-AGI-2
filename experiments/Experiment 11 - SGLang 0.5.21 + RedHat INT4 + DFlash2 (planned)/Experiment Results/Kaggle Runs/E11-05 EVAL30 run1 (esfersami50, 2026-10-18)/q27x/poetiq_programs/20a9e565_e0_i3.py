import numpy as np
from collections import deque


def transform(grid: np.ndarray) -> np.ndarray:
    """Reconstruct the small color strip indicated by the color-5 brackets.

    Rule (inferred from the examples):
      * The color-5 pixels form bracket markers.  Their bounding box gives the
        vertical extent (number of rows) of the answer.
      * The main figure is made of repeating small tiles of width ``w`` (2 or 3).
        The answer is a vertical strip ``out_h x w`` showing the figure's color
        "spine": a ladder (full row / side row alternating) in the figure's
        repeating colors.
      * The colors are taken from the figure's repeating color sequence.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()
    H, W = grid.shape

    def bbox(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        return int(r0), int(r1), int(c0), int(c1)

    b5 = bbox(grid == 5)
    if b5 is None:
        return np.zeros_like(grid)
    r0, r1, c0, c1 = b5
    out_h = r1 - r0 + 1

    def components(mask):
        Hm, Wm = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        comps = []
        for i in range(Hm):
            for j in range(Wm):
                if not mask[i, j] or seen[i, j]:
                    continue
                q = deque([(i, j)])
                seen[i, j] = True
                cells = []
                while q:
                    x, y = q.popleft()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < Hm and 0 <= ny < Wm and mask[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True
                            q.append((nx, ny))
                comps.append(cells)
        return comps

    colors = [int(x) for x in np.unique(grid) if x != 0 and x != 5]

    # ---- Detect tile width: 3x3 hollow rings (ladder family) vs 2x2 blocks ----
    ring = False
    if len(colors) > 0:
        for color in colors:
            for comp in components(grid == color):
                rmin = min(r for r, _ in comp); rmax = max(r for r, _ in comp)
                cmin = min(c for _, c in comp); cmax = max(c for _, c in comp)
                if (rmax - rmin + 1 == 3) and (cmax - cmin + 1 == 3):
                    sub = grid[rmin:rmax + 1, cmin:cmax + 1]
                    if sub[0, 1] == 0 and sub[1, 1] == 0 and sub[1, 0] == color and sub[1, 2] == color:
                        ring = True
                        break
            if ring:
                break

    # ---- Determine the two strip columns (x-offset, width) ----
    if ring:
        width, off = 3, 0
    else:
        width, off = 2, 1

    out_w = width

    # ---- Collect a color sequence for the repeating colors ----
    def comp_bbox(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        return min(rs), max(rs), min(cs), max(cs)

    seq = []
    # rightmost column of tiles left of the 5-bracket (used when the 5s sit
    # outside the figure, as in the diamond / triangle examples)
    if not ring:
        cands = []
        for color in colors:
            for comp in components(grid == color):
                ra, rb, ca, cb = comp_bbox(comp)
                if cb < c0 and ca >= c0 - 8:
                    cands.append((ca, cb, color, comp))
        if cands:
            cands.sort(key=lambda t: t[0])
            # take the right-most tile-column and read its colors top->bottom
            ca, cb, color0, comp0 = cands[-1]
            cells_sorted = sorted(comp0, key=lambda p: (p[0], p[1]))
            colseq = []
            for (r, cc) in cells_sorted:
                colseq.append(int(grid[r, cc]))
            # reduce to the repeating part
            seq = colseq
    if not seq:
        # fall back to left->right color of the distinct tile objects
        objs = []
        for color in colors:
            for comp in components(grid == color):
                ra, rb, ca, cb = comp_bbox(comp)
                objs.append((ca, color))
        objs.sort()
        seq = [c for _, c in objs]
    if not seq:
        seq = [colors[0]] if colors else [1]

    # ---- Choose the repeating color cycle and the starting color ----
    def detect_cycle(s):
        for L in range(1, len(s) + 1):
            ok = True
            for i in range(len(s) - L):
                if s[i] != s[i + L]:
                    ok = False
                    break
            if ok:
                return s[-L:]
        return s[-3:] if len(s) >= 3 else list(s)

    cycle = detect_cycle(seq)
    # starting color: prefer the color matching the figure's dominant spine
    if len(colors) > 0:
        counts = {}
        for color in colors:
            counts[color] = int(np.sum(grid == color))
        dominant = max(counts, key=counts.get)
    else:
        dominant = seq[0]
    if dominant in cycle:
        start = cycle.index(dominant)
    else:
        start = 0

    # ---- Build the ladder strip ----
    out = np.zeros((out_h, out_w), dtype=grid.dtype)
    for i in range(out_h):
        col = cycle[(start + i) % len(cycle)]
        if i % 2 == 0:
            out[i, :] = col
        else:
            out[i, 0] = col
            out[i, -1] = col

    return out