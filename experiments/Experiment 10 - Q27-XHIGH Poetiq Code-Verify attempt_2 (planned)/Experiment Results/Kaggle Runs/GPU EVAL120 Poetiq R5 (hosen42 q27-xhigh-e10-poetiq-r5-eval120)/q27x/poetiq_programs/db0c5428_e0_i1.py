import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a 9x9 embedded 3x3-tile pattern.

    The input is assumed to contain one 9x9 object made of nine 3x3 tiles.
    The central 3x3 tile is empty/background. The function:
      1. keeps the original 9x9 object,
      2. fills the central 3x3 tile with the eight inner cells from the
         surrounding tiles, using the base color for the middle cell,
      3. places 180-degree rotated copies of the eight surrounding tiles
         in the matching outer positions.
    """
    orig = np.array(grid, dtype=int, copy=True)

    if orig.ndim != 2:
        return orig

    out = orig.copy()
    H, W = orig.shape

    if H < 9 or W < 9:
        return out

    # Background color is the corner color in all examples.
    bg = orig[0, 0]

    # Locate the 9x9 object. First try the bounding box of non-background pixels.
    r0 = c0 = None
    nonbg = orig != bg
    if nonbg.any():
        rows = np.where(nonbg.any(axis=1))[0]
        cols = np.where(nonbg.any(axis=0))[0]
        if rows.size and cols.size:
            rr0, rr1 = int(rows[0]), int(rows[-1])
            cc0, cc1 = int(cols[0]), int(cols[-1])
            if rr1 - rr0 + 1 == 9 and cc1 - cc0 + 1 == 9:
                r0, c0 = rr0, cc0

    # Fallback: scan for a 9x9 window whose center 3x3 is background.
    if r0 is None:
        best_score = -1
        for r in range(H - 8):
            for c in range(W - 8):
                win = orig[r:r + 9, c:c + 9]

                # The central 3x3 tile must be empty/background.
                if not np.all(win[3:6, 3:6] == bg):
                    continue

                temp = win.copy()
                temp[3:6, 3:6] = bg
                score = int(np.sum(temp != bg))

                if score > best_score:
                    best_score = score
                    r0, c0 = r, c

        if r0 is None or best_score <= 0:
            return out

    obj = orig[r0:r0 + 9, c0:c0 + 9]

    # Determine the base color.
    # The cells immediately adjacent to the central hole are usually all base color.
    adj_coords = [
        (2, 3), (2, 5),
        (3, 2), (5, 2),
        (6, 3), (6, 5),
        (3, 6), (5, 6),
    ]
    adj_vals = [int(obj[dr, dc]) for dr, dc in adj_coords if obj[dr, dc] != bg]

    base = None
    if adj_vals and len(set(adj_vals)) == 1:
        base = adj_vals[0]

    # Fallback / tie-break: most frequent non-background color in the 9x9 object.
    obj_nonbg = obj[obj != bg]
    if obj_nonbg.size:
        vals, counts = np.unique(obj_nonbg, return_counts=True)
        overall_base = int(vals[np.argmax(counts)])
        if base is None or base == bg:
            base = overall_base
    else:
        base = bg

    if base == bg and obj_nonbg.size:
        vals, counts = np.unique(obj_nonbg, return_counts=True)
        base = int(vals[np.argmax(counts)])

    # Fill the central 3x3 tile.
    center = np.empty((3, 3), dtype=out.dtype)
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                center[i, j] = base
            else:
                # The eight key cells are at local rows/cols 2, 4, 6.
                center[i, j] = orig[r0 + 2 + 2 * i, c0 + 2 + 2 * j]

    out[r0 + 3:r0 + 6, c0 + 3:c0 + 6] = center

    # Add rotated copies of the eight surrounding 3x3 tiles.
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue

            src_r = r0 + 3 * i
            src_c = c0 + 3 * j

            tile = orig[src_r:src_r + 3, src_c:src_c + 3]
            rotated = tile[::-1, ::-1].copy()

            # Place the tile just outside the original object in the matching direction.
            sat_r = src_r + (i - 1) * 3
            sat_c = src_c + (j - 1) * 3

            if 0 <= sat_r and 0 <= sat_c and sat_r + 3 <= H and sat_c + 3 <= W:
                out[sat_r:sat_r + 3, sat_c:sat_c + 3] = rotated

    return out