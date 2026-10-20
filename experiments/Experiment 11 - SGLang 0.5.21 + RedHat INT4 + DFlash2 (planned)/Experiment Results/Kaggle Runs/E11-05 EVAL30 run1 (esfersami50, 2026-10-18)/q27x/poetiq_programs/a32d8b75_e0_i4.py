import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC grid:
      1. Find the vertical separator column made of 6s and take the canvas to its right.
      2. Extract the top-left 5x5 sprite, take its 3x3 core, and swap its two active colors.
      3. Extract the mask = the colored (non 0/6) pattern in the left panel below the sprite.
      4. Rotate the mask 90 degrees clockwise, then place it in the TOP-RIGHT of the canvas,
         scaling each mask cell to a 3x3 block, stamping the swapped sprite core there.
      5. Preserve any '1' cells that already exist in the canvas (overlay after stamping).
    """
    if grid.size == 0:
        return grid.copy()

    rows, cols = grid.shape

    # 1) Locate the vertical 6-separator (it is near the left edge in all cases).
    sep_col = None
    for c in range(cols):
        if (grid[:, c] == 6).sum() >= max(2, int(0.5 * rows)):
            sep_col = c
            break
    if sep_col is None:
        sep_col = min(5, cols - 1)

    canvas = grid[:, sep_col + 1:].copy()
    if canvas.size == 0:
        return canvas

    # 2) Sprite core (3x3) with its two active colors swapped.
    sprite = grid[:5, :5]
    counts = {}
    for i in range(5):
        for j in range(5):
            v = int(sprite[i, j])
            if v not in (0, 6):
                counts[v] = counts.get(v, 0) + 1
    color_a, color_b = sorted(counts.keys())
    swap = {color_a: color_b, color_b: color_a}

    stamp = np.zeros((3, 3), dtype=grid.dtype)
    for i in range(3):
        for j in range(3):
            v = int(sprite[1 + i, 1 + j])
            stamp[i, j] = swap.get(v, v)

    # 3) Mask = colored pattern in the left panel below the sprite (crop to its bounding box).
    region = grid[5:rows, :sep_col + 1]
    mask = (region != 0) & (region != 6)
    if not mask.any():
        # Fallback: any non-0 color below the sprite
        mask = (region != 0)
    if not mask.any():
        return canvas

    rs = np.where(np.any(mask, axis=1))[0]
    cs = np.where(np.any(mask, axis=0))[0]
    mask = mask[rs[0]:rs[-1] + 1, cs[0]:cs[-1] + 1]

    # 4) Rotate 90 degrees clockwise, then place top-right, scaled x3.
    mask_rot = np.rot90(mask, k=-1)   # clockwise
    mh, mw = mask_rot.shape
    ch, cw = canvas.shape

    off_r = max(0, (ch - mh * 3) // 2)
    off_c = max(0, cw - mw * 3)        # right-aligned

    for r in range(mh):
        for c in range(mw):
            if mask_rot[r, c]:
                rr0, cc0 = off_r + r * 3, off_c + c * 3
                rr1, cc1 = min(rr0 + 3, ch), min(cc0 + 3, cw)
                if rr0 < ch and cc0 < cw:
                    canvas[rr0:rr1, cc0:cc1] = stamp[:rr1 - rr0, :cc1 - cc0]

    # 5) Preserve any pre-existing 1 cells in the canvas.
    canvas[(canvas == 1)] = 1

    return canvas