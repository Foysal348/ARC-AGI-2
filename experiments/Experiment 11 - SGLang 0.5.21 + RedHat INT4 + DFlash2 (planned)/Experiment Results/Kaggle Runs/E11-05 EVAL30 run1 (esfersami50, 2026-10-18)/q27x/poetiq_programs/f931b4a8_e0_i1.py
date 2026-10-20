import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid according to the inferred ARC rule:

    - Split the grid into four equal quadrants.
    - Let B be the bottom-left quadrant (background).
    - Let A be the bottom-right quadrant (transparent overlay/stamp; 0 is transparent).
    - Let H be the number of nonzero cells in the top-left quadrant.
    - Let W be max(number of nonzero cells in the top-right quadrant, H).
    - Resize B to H x W using nearest-neighbor resizing.
    - Tile A at its native scale over the resized B.
    - Nonzero values in A override the background.

    Returns the transformed grid.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    qh, qw = h // 2, w // 2

    # If the grid is too small to split into quadrants, return it unchanged.
    if qh <= 0 or qw <= 0:
        return grid.copy()

    # Quadrants
    tl = grid[:qh, :qw]
    tr = grid[:qh, qw:]
    bl = grid[qh:, :qw]
    br = grid[qh:, qw:]

    # Determine target output dimensions from the top quadrants.
    target_h = int(np.count_nonzero(tl))
    target_w = int(np.count_nonzero(tr))

    # Graceful fallback for degenerate cases.
    if target_h <= 0:
        target_h = 1
    if target_w <= 0:
        target_w = target_h
    else:
        target_w = max(target_w, target_h)

    # Resize bottom-left background to target size.
    bh, bw = bl.shape
    row_idx = (np.arange(target_h) * bh) // target_h
    col_idx = (np.arange(target_w) * bw) // target_w
    background = bl[row_idx[:, None], col_idx[None, :]]

    # Tile bottom-right overlay at its original scale.
    ah, aw = br.shape
    tile_h = (target_h + ah - 1) // ah
    tile_w = (target_w + aw - 1) // aw
    overlay = np.tile(br, (tile_h, tile_w))[:target_h, :target_w]

    # Combine: nonzero overlay cells replace the background.
    output = background.copy()
    mask = overlay != 0
    output[mask] = overlay[mask]

    return output