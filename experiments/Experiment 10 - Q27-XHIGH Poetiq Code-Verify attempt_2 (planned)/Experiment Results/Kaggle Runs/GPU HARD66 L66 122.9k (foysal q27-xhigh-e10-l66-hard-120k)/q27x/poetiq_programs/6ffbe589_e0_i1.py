import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Isolate the main object, crop it, and rotate each color layer
    according to the color-specific rotation rule inferred from the examples.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.size == 0 or grid.ndim != 2:
        return grid.copy()

    nonzero = grid != 0
    if not nonzero.any():
        return grid.copy()

    def intervals(mask, gap=0):
        """
        Return True intervals in a 1D boolean array, merging intervals
        separated by at most `gap` False values.
        """
        raw = []
        in_int = False
        n = len(mask)

        for i, v in enumerate(mask):
            if v:
                if not in_int:
                    start = i
                    in_int = True
            else:
                if in_int:
                    raw.append((start, i - 1))
                    in_int = False

        if in_int:
            raw.append((start, n - 1))

        merged = []
        for s, e in raw:
            if merged and s - merged[-1][1] - 1 <= gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        return merged

    # Find the dominant column band of the main object.
    col_has = nonzero.any(axis=0)
    col_intervals = intervals(col_has, gap=1)
    if not col_intervals:
        return grid.copy()

    best_col = max(
        col_intervals,
        key=lambda t: (nonzero[:, t[0]:t[1] + 1].sum(), t[1] - t[0] + 1)
    )
    c0, c1 = best_col

    # Within that column band, find the dominant row band.
    sub_nz = nonzero[:, c0:c1 + 1]
    row_has = sub_nz.any(axis=1)
    row_intervals = intervals(row_has, gap=1)
    if not row_intervals:
        return grid.copy()

    best_row = max(
        row_intervals,
        key=lambda t: (sub_nz[t[0]:t[1] + 1, :].sum(), t[1] - t[0] + 1)
    )
    r0, r1 = best_row

    # Trim empty leading/trailing columns and rows from the selected band.
    region = grid[r0:r1 + 1, c0:c1 + 1]
    nz = region != 0

    col_ok = nz.any(axis=0)
    if not col_ok.any():
        return np.zeros_like(region)

    c_first = int(np.argmax(col_ok))
    c_last = int(len(col_ok) - 1 - np.argmax(col_ok[::-1]))
    region = region[:, c_first:c_last + 1]

    row_ok = (region != 0).any(axis=1)
    if not row_ok.any():
        return np.zeros_like(region)

    r_first = int(np.argmax(row_ok))
    r_last = int(len(row_ok) - 1 - np.argmax(row_ok[::-1]))
    crop = region[r_first:r_last + 1, :]

    def rotate_cw(arr, times):
        """Rotate a square 2D array clockwise by times * 90 degrees."""
        times %= 4
        if times == 0:
            return arr
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return arr
        # np.rot90 is counter-clockwise, so use k = 4 - times for clockwise.
        return np.rot90(arr, k=4 - times)

    # Rotation table inferred from the examples.
    angle_map = {
        1: 3,  # counter-clockwise
        2: 3,
        3: 1,  # clockwise
        4: 1,
        5: 0,
        6: 2,
        7: 2,
        8: 2,
        9: 0,
    }

    out = np.zeros(crop.shape, dtype=int)

    colors = np.unique(crop[crop != 0])
    if len(colors) == 0:
        return out

    counts = {int(c): int(np.count_nonzero(crop == c)) for c in colors}

    # Process inner/more sparse colors first, and the dominant color last,
    # so the outer layer has priority if rotations ever overlap.
    anchor = max(counts, key=lambda c: counts[c])
    order = [c for c in sorted(counts, key=lambda x: (counts[x], x)) if c != anchor]
    order.append(anchor)

    for color in order:
        mask = crop == color
        times = angle_map.get(int(color), 0)
        rotated = rotate_cw(mask, times)

        # Safety fallback: if rotation changed shape, keep original orientation.
        if rotated.shape != crop.shape:
            rotated = mask

        out[rotated] = int(color)

    return out