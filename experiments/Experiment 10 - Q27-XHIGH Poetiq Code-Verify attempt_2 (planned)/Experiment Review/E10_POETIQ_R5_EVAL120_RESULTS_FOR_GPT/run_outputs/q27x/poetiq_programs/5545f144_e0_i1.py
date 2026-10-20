import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an input grid containing one or more horizontally separated panels.

    Rule inferred from the examples:
    1. Split the input into equal-width panels at full vertical separator lines.
       A separator column is a constant column whose color is not the background.
       If no such separators exist, the whole grid is treated as one panel.
    2. In each panel, find rows that contain at least one non-background pixel.
       Keep only rows that are non-empty in every panel.
    3. Among those rows, choose the one closest to the top or bottom edge.
       Ties are resolved toward the top.
    4. In that chosen row, choose the most common column across panels.
       Ties are resolved by the column closest to the horizontal center,
       then by the smaller column index.
    5. Draw a small arrow-like marker on a blank panel-sized output:
       - odd number of panels: 3-cell marker
       - even number of panels: 4-cell marker
       Direction points toward the selected edge:
       - top: marker extends downward
       - bottom: marker extends upward
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Background is the most frequent color.
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])

    # ------------------------------------------------------------
    # Detect separator columns.
    # Separators are full columns of a non-background color.
    # They must divide the grid into equal-width panels.
    # ------------------------------------------------------------
    def has_equal_panel_widths(sep_cols):
        boundaries = [0] + list(sep_cols) + [W]
        widths = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            width = b - a - 1
            if width <= 0:
                return False
            widths.append(width)
        return all(w == widths[0] for w in widths)

    candidate_seps = []
    for c in range(W):
        col = grid[:, c]
        if col.min() == col.max() and int(col[0]) != bg:
            candidate_seps.append(c)

    sep_cols = None
    if candidate_seps:
        if has_equal_panel_widths(candidate_seps):
            sep_cols = candidate_seps
        else:
            # If multiple separator colors exist, try each constant-color group.
            groups = {}
            for c in candidate_seps:
                color = int(grid[0, c])
                groups.setdefault(color, []).append(c)

            for cols in groups.values():
                if has_equal_panel_widths(cols):
                    sep_cols = cols
                    break

    if sep_cols is None:
        panels = [grid]
    else:
        panels = []
        start = 0
        for c in sep_cols:
            panels.append(grid[:, start:c])
            start = c + 1
        panels.append(grid[:, start:W])

    # Basic sanity: all panels should have the same shape.
    ph, pw = panels[0].shape
    if any(p.shape != (ph, pw) for p in panels):
        # Fallback: no valid split, use whole grid.
        panels = [grid]
        ph, pw = grid.shape

    # ------------------------------------------------------------
    # Determine the object color: most frequent non-background color.
    # ------------------------------------------------------------
    obj_counts = {}
    for p in panels:
        nonbg = p[p != bg]
        if nonbg.size:
            vals, cnts = np.unique(nonbg, return_counts=True)
            for v, cnt in zip(vals, cnts):
                v = int(v)
                obj_counts[v] = obj_counts.get(v, 0) + int(cnt)

    if not obj_counts:
        return np.full((ph, pw), bg, dtype=grid.dtype)

    obj_color = max(obj_counts.items(), key=lambda x: x[1])[0]

    # ------------------------------------------------------------
    # Find rows that contain at least one object pixel in every panel.
    # ------------------------------------------------------------
    row_ok = np.ones(ph, dtype=bool)
    for p in panels:
        row_has = np.any(p != bg, axis=1)
        row_ok &= row_has

    candidate_rows = np.where(row_ok)[0]
    if candidate_rows.size == 0:
        return np.full((ph, pw), bg, dtype=grid.dtype)

    # Choose the candidate row closest to an edge; ties go to the top.
    best_r = int(min(candidate_rows, key=lambda r: (min(r, ph - 1 - r), r)))
    pointing_up = best_r <= ph - 1 - best_r

    # ------------------------------------------------------------
    # Choose the best column in that row.
    # Count, for each column, how many panels have an object there.
    # ------------------------------------------------------------
    col_counts = np.zeros(pw, dtype=int)
    for p in panels:
        mask = p[best_r] != bg
        # Count each panel at most once per column.
        col_counts[mask] += 1

    max_col_count = int(col_counts.max())
    if max_col_count == 0:
        best_c = pw // 2
    else:
        cand_cols = np.where(col_counts == max_col_count)[0]
        center = (pw - 1) / 2.0
        best_c = int(min(cand_cols.tolist(), key=lambda c: (abs(c - center), c)))

    # ------------------------------------------------------------
    # Draw the marker.
    # ------------------------------------------------------------
    out = np.full((ph, pw), bg, dtype=grid.dtype)

    def put(r, c):
        if 0 <= r < ph and 0 <= c < pw:
            out[r, c] = obj_color

    n_panels = len(panels)

    if n_panels % 2 == 1:
        # 3-cell marker.
        if pointing_up:
            put(best_r, best_c)
            put(best_r + 1, best_c - 1)
            put(best_r + 1, best_c + 1)
        else:
            put(best_r, best_c)
            put(best_r - 1, best_c - 1)
            put(best_r - 1, best_c + 1)
    else:
        # 4-cell marker.
        if pointing_up:
            put(best_r, best_c)
            put(best_r + 1, best_c)
            put(best_r + 2, best_c - 1)
            put(best_r + 2, best_c + 1)
        else:
            put(best_r, best_c)
            put(best_r - 1, best_c - 1)
            put(best_r - 1, best_c)
            put(best_r - 1, best_c + 1)

    return out