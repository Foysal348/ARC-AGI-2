import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Repair small broken straight line segments.

    The grid has one background color and one object color.  A gap is a short
    background segment between two object-color runs in the same row or column.

    A gap is repaired when:
      * its length is 1, 2, or 3;
      * the run before the gap (left for rows, above for columns) is strictly
        longer than the run after the gap;
      * for gaps longer than 1, the run after the gap must be a "real"
        endpoint: it has length at least 2, or it touches the border in the
        forward direction.

    While repairing a gap:
      * if a matching object pixel has been displaced perpendicularly by 1 or 2
        cells (down for horizontal gaps, right for vertical gaps), move that
        pixel back into the gap;
      * otherwise fill the missing gap cell with the object color.

    A displaced pixel is only moved if it is locally isolated in the
    perpendicular direction, so pixels that belong to another nearby line are
    not stolen.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    out = grid.copy()
    H, W = grid.shape

    # Identify background and the main object color.
    vals, counts = np.unique(grid, return_counts=True)
    background = int(vals[np.argmax(counts)])
    fg = (grid != background)

    if not fg.any():
        return out

    nonbg = grid[fg]
    obj_vals, obj_counts = np.unique(nonbg, return_counts=True)
    object_color = int(obj_vals[np.argmax(obj_counts)])

    used = set()
    ISO_WINDOW = 5

    def is_isolated(r, c, check_column: bool) -> bool:
        """
        Return True if a candidate displaced pixel looks isolated.

        For a horizontal line gap, check the same column.
        For a vertical line gap, check the same row.
        """
        if check_column:
            r0 = max(0, r - ISO_WINDOW)
            r1 = min(H - 1, r + ISO_WINDOW)
            for rr in range(r0, r1 + 1):
                if rr != r and fg[rr, c]:
                    return False
        else:
            c0 = max(0, c - ISO_WINDOW)
            c1 = min(W - 1, c + ISO_WINDOW)
            for cc in range(c0, c1 + 1):
                if cc != c and fg[r, cc]:
                    return False
        return True

    def find_row_gap_stray(r, c):
        """Find a displaced pixel below a horizontal gap cell."""
        for d in (1, 2):
            rr = r + d
            if rr < H and fg[rr, c] and (rr, c) not in used:
                if is_isolated(rr, c, True):
                    return (rr, c)
        return None

    def find_col_gap_stray(c, r):
        """Find a displaced pixel to the right of a vertical gap cell."""
        for d in (1, 2):
            cc = c + d
            if cc < W and fg[r, cc] and (r, cc) not in used:
                if is_isolated(r, cc, False):
                    return (r, cc)
        return None

    row_gaps = []
    col_gaps = []

    # Find eligible horizontal gaps.
    for r in range(H):
        runs = []
        c = 0
        while c < W:
            if fg[r, c]:
                start = c
                while c < W and fg[r, c]:
                    c += 1
                end = c - 1
                runs.append((start, end, end - start + 1))
            else:
                c += 1

        for i in range(len(runs) - 1):
            _, e1, l1 = runs[i]
            s2, e2, l2 = runs[i + 1]
            g_start = e1 + 1
            g_end = s2 - 1
            g_len = g_end - g_start + 1

            if 1 <= g_len <= 3 and l1 > l2:
                trailing_at_border = (e2 == W - 1)
                if g_len == 1 or l2 >= 2 or trailing_at_border:
                    row_gaps.append((r, g_start, g_end))

    # Find eligible vertical gaps.
    for c in range(W):
        runs = []
        r = 0
        while r < H:
            if fg[r, c]:
                start = r
                while r < H and fg[r, c]:
                    r += 1
                end = r - 1
                runs.append((start, end, end - start + 1))
            else:
                r += 1

        for i in range(len(runs) - 1):
            _, e1, l1 = runs[i]
            s2, _, l2 = runs[i + 1]
            g_start = e1 + 1
            g_end = s2 - 1
            g_len = g_end - g_start + 1

            if 1 <= g_len <= 3 and l1 > l2:
                trailing_at_border = (e1 == H - 1)
                if g_len == 1 or l2 >= 2 or trailing_at_border:
                    col_gaps.append((c, g_start, g_end))

    # Repair horizontal gaps.
    for r, c_start, c_end in row_gaps:
        for c in range(c_start, c_end + 1):
            stray = find_row_gap_stray(r, c)
            if stray is not None:
                rr, cc = stray
                out[rr, cc] = background
                out[r, c] = grid[rr, cc]
                used.add((rr, cc))
            else:
                out[r, c] = object_color

    # Repair vertical gaps.
    for c, r_start, r_end in col_gaps:
        for r in range(r_start, r_end + 1):
            stray = find_col_gap_stray(c, r)
            if stray is not None:
                rr, cc = stray
                out[rr, cc] = background
                out[r, c] = grid[rr, cc]
                used.add((rr, cc))
            else:
                out[r, c] = object_color

    return out