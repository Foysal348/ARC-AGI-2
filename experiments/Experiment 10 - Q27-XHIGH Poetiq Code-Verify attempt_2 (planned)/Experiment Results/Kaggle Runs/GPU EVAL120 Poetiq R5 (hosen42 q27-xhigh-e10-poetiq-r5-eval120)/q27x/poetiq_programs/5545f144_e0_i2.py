import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid with one or more equal-width panels separated by full
    vertical separator columns.

    Rule:
      - Split the grid at valid vertical separator columns.
      - Output is one blank panel-sized grid.
      - Find the row that has the object color in every panel and is closest
        to the top or bottom edge; direction points to that edge.
      - In that row, choose the most common column across panels, with ties
        resolved toward the horizontal center, then leftmost.
      - Draw a small arrow pointing to that selected row/column.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    if grid.ndim != 2:
        grid = np.atleast_2d(grid)

    H, W = grid.shape

    # Background is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Candidate separator columns: full constant columns, not background.
    candidate_seps = []
    for c in range(W):
        col = grid[:, c]
        if col.min() == col.max() and int(col[0]) != bg:
            candidate_seps.append(c)

    def valid_panel_width(seps):
        """Return panel width if these separators make equal-width panels."""
        seps = sorted(seps)
        bounds = [0] + seps + [W]
        widths = [b - a - 1 for a, b in zip(bounds[:-1], bounds[1:])]

        if any(w <= 0 for w in widths):
            return None

        if all(w == widths[0] for w in widths):
            return widths[0]

        return None

    sep_cols = None
    if candidate_seps:
        # First try all candidate separator columns together.
        pw = valid_panel_width(candidate_seps)
        if pw is not None:
            sep_cols = candidate_seps
        else:
            # If that fails, try each constant-color group.
            groups = {}
            for c in candidate_seps:
                color = int(grid[0, c])
                groups.setdefault(color, []).append(c)

            for cols in sorted(groups.values(), key=lambda x: len(x), reverse=True):
                pw = valid_panel_width(cols)
                if pw is not None:
                    sep_cols = cols
                    break

    if sep_cols is None:
        panels = [grid]
        panel_h, panel_w = H, W
    else:
        panels = []
        start = 0
        for c in sep_cols:
            panels.append(grid[:, start:c])
            start = c + 1
        panels.append(grid[:, start:W])

        # Safety fallback if something unexpected happened.
        if any(p.shape != (H, panels[0].shape[1]) for p in panels):
            panels = [grid]
            panel_h, panel_w = H, W
        else:
            panel_h, panel_w = H, panels[0].shape[1]

    # Main object color: most frequent non-background color inside panels.
    obj_counts = {}
    for p in panels:
        mask = p != bg
        if mask.any():
            v, cnt = np.unique(p[mask], return_counts=True)
            for color, c in zip(v, cnt):
                obj_counts[int(color)] = obj_counts.get(int(color), 0) + int(c)

    if not obj_counts:
        return np.full((panel_h, panel_w), bg, dtype=grid.dtype)

    obj = max(obj_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    # Count, for each row, how many panels contain the object color in that row.
    row_counts = np.zeros(H, dtype=int)
    for r in range(H):
        cnt = 0
        for p in panels:
            if np.any(p[r] == obj):
                cnt += 1
        row_counts[r] = cnt

    # Preferred rows: present in every panel. Fallback: most common row count.
    n_panels = len(panels)
    candidate_rows = np.where(row_counts == n_panels)[0]
    if candidate_rows.size == 0:
        max_rc = int(row_counts.max())
        if max_rc == 0:
            return np.full((panel_h, panel_w), bg, dtype=grid.dtype)
        candidate_rows = np.where(row_counts == max_rc)[0]

    # Choose row closest to a horizontal edge; tie goes to top.
    best_r = min(candidate_rows.tolist(), key=lambda r: (min(r, H - 1 - r), r))
    pointing_up = best_r <= H - 1 - best_r

    # In the chosen row, choose the most common column across panels.
    col_counts = np.zeros(panel_w, dtype=int)
    for p in panels:
        col_counts += (p[best_r] == obj).astype(int)

    if col_counts.max() == 0:
        best_c = panel_w // 2
    else:
        max_cc = int(col_counts.max())
        cand_cols = np.where(col_counts == max_cc)[0]
        center = (panel_w - 1) / 2.0
        best_c = min(cand_cols.tolist(), key=lambda c: (abs(c - center), c))

    # Keep the arrow inside the panel horizontally.
    if panel_w >= 3:
        best_c = min(max(best_c, 1), panel_w - 2)
    else:
        best_c = 0

    out = np.full((panel_h, panel_w), bg, dtype=grid.dtype)

    def put(r, c):
        if 0 <= r < panel_h and 0 <= c < panel_w:
            out[r, c] = obj

    if pointing_up:
        if best_r == 0 and H >= 3:
            # Long upward arrow when the selected row is the top edge.
            cells = [
                (best_r, best_c),
                (best_r + 1, best_c),
                (best_r + 2, best_c - 1),
                (best_r + 2, best_c + 1),
            ]
        else:
            # Short upward arrow.
            cells = [
                (best_r, best_c),
                (best_r + 1, best_c - 1),
                (best_r + 1, best_c + 1),
            ]
    else:
        if best_r == H - 1 and H >= 3:
            # Long downward arrow when the selected row is the bottom edge.
            cells = [
                (best_r, best_c),
                (best_r - 1, best_c),
                (best_r - 2, best_c - 1),
                (best_r - 2, best_c + 1),
            ]
        else:
            # Short downward arrow.
            cells = [
                (best_r, best_c),
                (best_r - 1, best_c - 1),
                (best_r - 1, best_c),
                (best_r - 1, best_c + 1),
            ]

    for r, c in cells:
        put(r, c)

    return out