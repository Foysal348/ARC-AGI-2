import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:
    1. Identify the two main non-background colors.
       One of them is treated as the "primary" color. It is chosen as the
       color of the highest non-background pixel (topmost, then leftmost).
       This matches the examples, where the top-left/topmost major color
       keeps its protrusions into the other color, while the other color's
       protrusions into it are removed.

    2. Clean single-pixel noise using a directional local rule:
       - If a pixel has color primary and 3 of its 4 orthogonal neighbors
         are background (0), it is removed (set to 0).
       - If a pixel has color secondary and 3 of its 4 orthogonal neighbors
         are primary, it is changed to primary.
       - If a pixel has color secondary and 3 of its 4 orthogonal neighbors
         are background, it is removed (set to 0).

    3. Find "holes": 0-pixels that have at least 3 orthogonal neighbors of
       the same non-zero color C. For each such hole, draw a 3x3 motif:
       - center becomes 8
       - the 8 surrounding cells become the other main color (not C).

    4. Motifs are drawn after cleaning. Centers always overwrite fill colors.
    """
    grid = np.array(grid, copy=True)
    if grid.size == 0:
        return grid

    h, w = grid.shape

    # Main colors: ignore background 0 and output marker 8 if present.
    colors = [c for c in np.unique(grid) if c not in (0, 8)]
    if len(colors) < 2:
        # Fallback: no meaningful two-color pattern.
        return grid

    # Choose primary color as the color of the first non-background pixel
    # in row-major order.
    primary = None
    for r in range(h):
        for c in range(w):
            v = grid[r, c]
            if v in colors:
                primary = int(v)
                break
        if primary is not None:
            break

    if primary not in colors:
        return grid
    secondary = int(colors[0] if colors[0] != primary else colors[1])
    other = {primary: secondary, secondary: primary}

    def neighbors(r, c):
        vals = []
        if r > 0:
            vals.append(int(grid[r - 1, c]))
        if r + 1 < h:
            vals.append(int(grid[r + 1, c]))
        if c > 0:
            vals.append(int(grid[r, c - 1]))
        if c + 1 < w:
            vals.append(int(grid[r, c + 1]))
        return vals

    # 1) Clean noise.
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            if v not in (primary, secondary):
                continue
            nb = neighbors(r, c)
            zero_count = sum(1 for x in nb if x == 0)
            prim_count = sum(1 for x in nb if x == primary)
            sec_count = sum(1 for x in nb if x == secondary)

            if v == primary:
                # Primary protrusions into background are removed.
                if zero_count >= 3:
                    grid[r, c] = 0
            else:  # v == secondary
                # Secondary intrusions into primary are changed to primary.
                if prim_count >= 3:
                    grid[r, c] = primary
                # Secondary protrusions into background are removed.
                elif zero_count >= 3:
                    grid[r, c] = 0

    # 2) Find holes and record motif centers/fills.
    centers = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0:
                continue
            nb = neighbors(r, c)
            for color in (primary, secondary):
                if sum(1 for x in nb if x == color) >= 3:
                    centers.append((r, c, color))
                    break

    out = grid.copy()

    # 3) Draw 3x3 motifs.
    for r, c, color in centers:
        fill = other[color]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not (dr == 0 and dc == 0):
                    out[rr, cc] = fill

    # 4) Place centers on top.
    for r, c, _ in centers:
        out[r, c] = 8

    return out