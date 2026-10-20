import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange separator-delimited panels into a single row or column.

    The input grid is split by full rows/columns of the separator color.
    In the provided examples the separator color is 6. The panels are then
    ordered by diagonal distance from the main diagonal. If the panel grid is
    taller than it is wide, the panels are written in a single horizontal row
    and the diagonal order is reversed. Otherwise they are written in a single
    vertical column.

    This version fixes separator detection: it no longer treats an object line
    (for example, a full column of color 1 inside one panel) as the separator
    when a valid color-6 separator lattice exists.
    """

    def intervals_from_seps(seps, length):
        """Return non-empty intervals between full separator lines."""
        bounds = [-1] + list(seps) + [length]
        intervals = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            start = a + 1
            end = b
            if end > start:
                intervals.append((start, end))
        return intervals

    def candidate_info(g, color):
        """
        Assess whether a color can be the separator color.

        A good separator color should occur on full rows/columns, preferably
        internal ones, and should not have many stray pixels outside those
        full lines.
        """
        h, w = g.shape
        mask = g == color

        full_rows = np.flatnonzero(np.all(mask, axis=1)).tolist()
        full_cols = np.flatnonzero(np.all(mask, axis=0)).tolist()

        if not full_rows and not full_cols:
            return None

        union = np.zeros((h, w), dtype=bool)
        if full_rows:
            union[full_rows, :] = True
        if full_cols:
            union[:, full_cols] = True

        extra = int(np.any(mask & ~union))

        row_iv = intervals_from_seps(full_rows, h)
        col_iv = intervals_from_seps(full_cols, w)

        if not row_iv or not col_iv:
            return None

        row_lengths = [end - start for start, end in row_iv]
        col_lengths = [end - start for start, end in col_iv]

        uniform = (
            len(set(row_lengths)) <= 1 and
            len(set(col_lengths)) <= 1
        )

        internal_rows = sum(1 for r in full_rows if 0 < r < h - 1)
        internal_cols = sum(1 for c in full_cols if 0 < c < w - 1)

        return {
            "extra": extra,
            "internal_rows": internal_rows,
            "internal_cols": internal_cols,
            "line_count": len(full_rows) + len(full_cols),
            "uniform": uniform,
            "panel_count": len(row_iv) * len(col_iv),
        }

    def make_score(info):
        """Lower score is better."""
        return (
            info["extra"],
            -(info["internal_rows"] + info["internal_cols"]),
            -info["line_count"],
            0 if info["uniform"] else 1,
            -info["panel_count"],
        )

    def find_separator_color(g):
        """
        Find the separator color.

        The examples use color 6. If 6 forms a valid separator structure,
        prefer it immediately. This prevents object lines such as a full
        column of color 1 from being mistaken for the separator.
        """
        info6 = candidate_info(g, 6)
        if info6 is not None:
            return 6

        if g.size == 0:
            return None

        values, counts = np.unique(g, return_counts=True)
        background = int(values[np.argmax(counts)])

        best_score = None
        best_color = None

        # First try colors that are not the global background color.
        for value in values:
            color = int(value)
            if color == background:
                continue

            info = candidate_info(g, color)
            if info is None:
                continue

            score = make_score(info)
            if best_score is None or score < best_score:
                best_score = score
                best_color = color

        if best_color is not None:
            return best_color

        # Fallback: allow background color only if no other candidate exists.
        for value in values:
            color = int(value)
            info = candidate_info(g, color)
            if info is None:
                continue

            score = make_score(info)
            if best_score is None or score < best_score:
                best_score = score
                best_color = color

        return best_color

    def get_background(g, sep):
        """Most common non-separator color, used for padding if needed."""
        non_sep = g[g != sep]
        if non_sep.size == 0:
            return 0
        vals, counts = np.unique(non_sep, return_counts=True)
        return int(vals[np.argmax(counts)])

    if grid is None:
        return np.array([], dtype=int)

    g = np.asarray(grid, dtype=int)

    if g.ndim != 2 or g.size == 0:
        return g.copy()

    h, w = g.shape
    sep = find_separator_color(g)

    if sep is None:
        return g.copy()

    row_seps = [int(r) for r in np.flatnonzero(np.all(g == sep, axis=1))]
    col_seps = [int(c) for c in np.flatnonzero(np.all(g == sep, axis=0))]

    if not row_seps and not col_seps:
        return g.copy()

    row_intervals = intervals_from_seps(row_seps, h)
    col_intervals = intervals_from_seps(col_seps, w)

    if not row_intervals or not col_intervals:
        return g.copy()

    panels = [
        [g[r0:r1, c0:c1].copy() for c0, c1 in col_intervals]
        for r0, r1 in row_intervals
    ]

    R = len(panels)
    C = len(panels[0]) if R else 0

    if R == 0 or C == 0:
        return g.copy()

    # Diagonal order: distance from main diagonal first, then anti-diagonal sum.
    cells = [(r, c) for r in range(R) for c in range(C)]
    cells.sort(key=lambda rc: (abs(rc[0] - rc[1]), rc[0] + rc[1], rc[0], rc[1]))

    # If the panel grid is taller than wide, make a horizontal strip and
    # reverse the diagonal order. Otherwise make a vertical strip.
    if R > C:
        order = cells[::-1]
        horizontal = True
    else:
        order = cells
        horizontal = False

    max_panel_h = max(p.shape[0] for row in panels for p in row)
    max_panel_w = max(p.shape[1] for row in panels for p in row)
    bg = get_background(g, sep)

    def pad_panel(p):
        if p.shape == (max_panel_h, max_panel_w):
            return p
        padded = np.full((max_panel_h, max_panel_w), bg, dtype=g.dtype)
        padded[:p.shape[0], :p.shape[1]] = p
        return padded

    n = len(order)

    if n == 1:
        r, c = order[0]
        return pad_panel(panels[r][c])

    if horizontal:
        out = np.full(
            (max_panel_h, n * max_panel_w + (n - 1)),
            bg,
            dtype=g.dtype,
        )

        x = 0
        for i, (r, c) in enumerate(order):
            panel = pad_panel(panels[r][c])
            out[:, x:x + max_panel_w] = panel
            x += max_panel_w

            if i < n - 1:
                out[:, x] = sep
                x += 1

        return out

    out = np.full(
        (n * max_panel_h + (n - 1), max_panel_w),
        bg,
        dtype=g.dtype,
    )

    y = 0
    for i, (r, c) in enumerate(order):
        panel = pad_panel(panels[r][c])
        out[y:y + max_panel_h, :] = panel
        y += max_panel_h

        if i < n - 1:
            out[y, :] = sep
            y += 1

    return out