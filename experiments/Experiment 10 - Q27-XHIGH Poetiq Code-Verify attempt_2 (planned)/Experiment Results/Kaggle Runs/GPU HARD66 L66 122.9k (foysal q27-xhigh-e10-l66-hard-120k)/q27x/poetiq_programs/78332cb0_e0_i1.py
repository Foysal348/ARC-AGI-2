import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange 5x5 tiles separated by lines of 6.

    Rules:
    - Single row of tiles -> single column, same left-to-right order.
    - Single column of tiles -> single row, reverse top-to-bottom order.
    - 2D arrangement of tiles -> single column, sorted by tile size
      (number of pixels that are not 7), with row-major order as tie-break.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    SEP = 6
    BG = 7

    # Find full separator rows and columns.
    h_seps = [r for r in range(H) if np.all(grid[r, :] == SEP)]
    v_seps = [c for c in range(W) if np.all(grid[:, c] == SEP)]

    def make_bounds(seps, length):
        bounds = []
        prev = 0
        for s in seps:
            if s > prev:
                bounds.append((prev, s))
            prev = s + 1
        if prev < length:
            bounds.append((prev, length))
        return bounds

    row_bounds = make_bounds(h_seps, H) or [(0, H)]
    col_bounds = make_bounds(v_seps, W) or [(0, W)]

    # Extract tiles and count their non-background pixels.
    blocks = []
    for ri, (r0, r1) in enumerate(row_bounds):
        for ci, (c0, c1) in enumerate(col_bounds):
            tile = grid[r0:r1, c0:c1]
            size = int(np.count_nonzero(tile != BG))
            blocks.append((ri, ci, size, tile))

    R = len(row_bounds)
    C = len(col_bounds)

    if R == 1:
        order = blocks
        vertical = True
    elif C == 1:
        order = blocks[::-1]
        vertical = False
    else:
        # Stable sort preserves row-major order for equal sizes.
        order = sorted(blocks, key=lambda x: x[2])
        vertical = True

    if not order:
        return np.zeros_like(grid)

    h = order[0][3].shape[0]
    w = order[0][3].shape[1]
    n = len(order)

    if vertical:
        out = np.full((h * n + (n - 1), w), BG, dtype=grid.dtype)
        for i, (_, _, _, tile) in enumerate(order):
            r = i * (h + 1)
            out[r:r + h, :] = tile
            if i != n - 1:
                out[r + h, :] = SEP
    else:
        out = np.full((h, w * n + (n - 1)), BG, dtype=grid.dtype)
        for i, (_, _, _, tile) in enumerate(order):
            c = i * (w + 1)
            out[:, c:c + w] = tile
            if i != n - 1:
                out[:, c + w] = SEP

    return out