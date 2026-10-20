import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid into a compact ranking chart.

    Rule inferred from the examples:
    - The input contains several structured colored "boxes"/grids, plus one
      distractor/noise color.
    - For each structured color, estimate how many cells its grid contains.
      A cell count of at least 4 is displayed as 4 because the output width
      is fixed to 4.
    - Sort the structured colors by cell count (ascending).  Ties are broken
      by top position, then left position, then color.
    - Create an output grid with one row per structured color and 4 columns.
      In row i, the first min(cell_count, 4) cells are that color, and the
      remaining cells are filled with the noise color.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    colors = set(int(c) for c in grid.flat if c != 0)

    def max_run(mask_row: np.ndarray) -> int:
        best = cur = 0
        for x in mask_row:
            if x:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best

    # Score each non-background color by how "line-like" it is.
    scores = {}
    for c in colors:
        mask = (grid == c)
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        if not rows.any() or not cols.any():
            scores[c] = 0
            continue
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        max_horiz = max((int(max_run(mask[r, c0:c1 + 1])) for r in range(r0, r1 + 1)), default=0)
        max_vert = max((int(max_run(mask[r0:r1 + 1, c])) for c in range(c0, c1 + 1)), default=0)

        # A structured object must have at least one reasonably long line.
        score = max(max_horiz, max_vert)
        if score >= 3:
            score += min(100, int(np.count_nonzero(mask)))
        scores[c] = score

    # The noise/filler color is the color with the lowest structural score.
    filler = min(colors, key=lambda c: scores[c])

    # Object colors are the structured ones.
    object_colors = [c for c in colors if c != filler and scores[c] >= 3]

    # If the above heuristic misses something, fall back to the non-noise colors.
    if not object_colors:
        object_colors = [c for c in colors if c != filler]

    def estimate_cell_count(c: int) -> int:
        """
        Estimate the number of grid cells for a colored structured object.

        The estimate is based on detecting vertical/horizontal grid lines
        inside the object's bounding box.  Occlusion/noise is tolerated by
        using occupancy thresholds rather than perfect pixel counts.
        """
        mask = (grid == c)
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        if not rows.any() or not cols.any():
            return 1

        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        height = r1 - r0 + 1
        width = c1 - c0 + 1

        if height < 3 or width < 3:
            return 1

        # Vertical lines: columns with many pixels of this color in interior rows.
        interior_v = mask[r0 + 1:r1, c0:c1 + 1]
        if interior_v.shape[0] > 0:
            col_counts = interior_v.sum(axis=0)
            v_thr = max(1, int(0.35 * interior_v.shape[0]))
            # Also require at least two pixels for sufficiently tall objects.
            if interior_v.shape[0] >= 4:
                v_thr = max(2, v_thr)
            v_lines = int(np.count_nonzero(col_counts >= v_thr))
        else:
            v_lines = 1

        # Horizontal lines: rows with many pixels of this color in interior columns.
        interior_h = mask[r0:r1 + 1, c0 + 1:c1]
        if interior_h.shape[1] > 0:
            row_counts = interior_h.sum(axis=1)
            h_thr = max(1, int(0.35 * interior_h.shape[1]))
            if interior_h.shape[1] >= 4:
                h_thr = max(2, h_thr)
            h_lines = int(np.count_nonzero(row_counts >= h_thr))
        else:
            h_lines = 1

        # If the object is clearly a simple rectangle, force at least one cell.
        cells = max(1, (v_lines - 1) * (h_lines - 1))

        # In all examples, the displayed bar length is capped at 4.
        return min(cells, 4)

    items = []
    for c in object_colors:
        mask = (grid == c)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        top = int(rows[0])
        left = int(cols[0])
        size = estimate_cell_count(c)
        items.append((size, top, left, c))

    # Sort by size, then position, then color.
    items.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

    n = len(items)
    if n == 0:
        # Should not happen for valid examples; return an empty grid.
        return np.empty((0, 4), dtype=np.uint8)

    out = np.full((n, 4), filler, dtype=np.uint8)
    for i, (size, _, _, c) in enumerate(items):
        out[i, :min(size, 4)] = c

    return out