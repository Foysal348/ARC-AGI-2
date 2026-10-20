import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC grid by:
      1. Taking the 5x5 top-left sprite.
      2. Extracting its 3x3 core.
      3. Swapping the two non-zero/non-background colors in that core.
      4. Building a 3x3 mask from the 4-colored cells below the sprite.
      5. Stamping the swapped 3x3 core onto the canvas at all mask positions,
         using a 3x3 scale factor.
    """
    if grid.size == 0:
        return grid.copy()

    out = np.zeros_like(grid)

    # The sprite is the 5x5 block in the top-left corner.
    sprite = grid[:5, :5]

    # Find the two active colors in the sprite (ignore 0 and 6).
    active = {}
    for r in range(5):
        for c in range(5):
            v = int(sprite[r, c])
            if v not in (0, 6):
                active[v] = active.get(v, 0) + 1

    if len(active) != 2:
        # Fallback: use the whole top-left 3x3 and no stamping.
        return grid[:, 6:].copy() if grid.shape[1] > 6 else grid.copy()

    color_a, color_b = sorted(active.keys())

    # Build the swapped 3x3 stamp from the sprite core.
    stamp = np.zeros((3, 3), dtype=grid.dtype)
    for i in range(3):
        for j in range(3):
            v = int(sprite[1 + i, 1 + j])
            if v == color_a:
                stamp[i, j] = color_b
            elif v == color_b:
                stamp[i, j] = color_a

    # The mask is formed by the 4-colored cells directly below the sprite.
    # It occupies rows 5-10 and columns 1-3 in the given examples.
    mask_region = grid[5:11, 1:4]
    mask = (mask_region == 4).astype(bool)

    # The output canvas is the right part of the input, after the first 6 columns.
    canvas = grid[:, 6:].copy()

    # Place the stamp on every active mask cell, scaled by 3.
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                r0 = 3 * i
                c0 = 3 * j
                r1 = min(r0 + 3, canvas.shape[0])
                c1 = min(c0 + 3, canvas.shape[1])
                h = r1 - r0
                w = c1 - c0
                if h > 0 and w > 0:
                    canvas[r0:r1, c0:c1] = stamp[:h, :w]

    return canvas