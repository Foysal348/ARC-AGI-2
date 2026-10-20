import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:
    - Find the vertical separator column and the empty rectangular canvas to its right.
    - Identify the frame background color.
    - Treat non-background colors on the frame sides as row/column markers.
    - For each color with both row and column markers, erase the matching 8-connected
      component(s) in the left margin and stamp it at all marked row/column intersections.
    - Stamps are clipped to the interior canvas.
    - Overlaps are resolved by drawing bottom row markers first and top row markers last.
    """
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        return g
    H, W = g.shape
    if H < 3 or W < 5:
        return g.copy()

    def constant_nonzero_col(c):
        if c < 0 or c >= W:
            return False
        col = g[:, c]
        return bool(np.all(col == col[0])) and int(col[0]) != 0

    # ------------------------------------------------------------------
    # Detect separator column.
    # The interior is normally a zero block bounded by non-zero frame columns.
    # If the interior starts at column s, the separator is at s - 2.
    # ------------------------------------------------------------------
    sep_col = None
    if H > 2:
        zero_col = np.all(g[1:-1, :] == 0, axis=0)
        runs = []
        c = 0
        while c < W:
            if zero_col[c]:
                s = c
                while c < W and zero_col[c]:
                    c += 1
                runs.append((s, c - 1))
            else:
                c += 1

        candidates = []
        for s, e in runs:
            if s >= 2 and e < W - 1:
                csep = s - 2
                if constant_nonzero_col(csep):
                    # The column between separator and interior is the left frame side.
                    if g[0, csep + 1] != 0 and g[H - 1, csep + 1] != 0:
                        candidates.append((e - s + 1, csep))

        if candidates:
            # Prefer the rightmost valid separator, then the widest interior.
            candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
            sep_col = candidates[0][1]

    # Fallback: find a full-height non-zero constant column with an empty area to its right.
    if sep_col is None:
        for c in range(1, W - 1):
            if constant_nonzero_col(c) and np.any(g[:, :c] != 0):
                if c + 2 < W - 1 and H > 2:
                    size = (H - 2) * (W - c - 3)
                    if size > 0 and np.count_nonzero(g[1:-1, c + 2:-2]) / size < 0.1:
                        sep_col = c
                        break
        if sep_col is None:
            for c in range(1, W - 1):
                if constant_nonzero_col(c):
                    sep_col = c
                    break

    if sep_col is None or sep_col + 2 >= W - 1:
        return g.copy()

    sep_val = int(g[0, sep_col])
    left_frame = sep_col + 1
    right_frame = W - 1
    inner_left = left_frame + 1
    inner_right = right_frame - 1

    if inner_left > inner_right:
        return g.copy()

    # ------------------------------------------------------------------
    # Frame background color: most frequent color on frame edges.
    # ------------------------------------------------------------------
    frame_vals = np.concatenate((
        g[0, left_frame:right_frame + 1].ravel(),
        g[H - 1, left_frame:right_frame + 1].ravel(),
        g[0:H, left_frame].ravel(),
        g[0:H, right_frame].ravel()
    ))

    if frame_vals.size == 0:
        return g.copy()

    base = int(np.argmax(np.bincount(frame_vals)))
    if base == 0:
        nz = frame_vals[frame_vals != 0]
        if nz.size:
            vals, counts = np.unique(nz, return_counts=True)
            base = int(vals[np.argmax(counts)])

    # ------------------------------------------------------------------
    # Collect frame markers.
    # Row markers come from left/right frame sides.
    # Column markers come from top/bottom frame sides.
    # ------------------------------------------------------------------
    row_marks = {}
    col_marks = {}

    def add_row(val, r):
        val = int(val)
        if val != 0 and val != sep_val and val != base:
            row_marks.setdefault(val, set()).add(r)

    def add_col(val, c):
        val = int(val)
        if val != 0 and val != sep_val and val != base:
            col_marks.setdefault(val, set()).add(c)

    for r in range(H):
        for c in (left_frame, right_frame):
            add_row(g[r, c], r)

    for c in range(left_frame, right_frame + 1):
        for r in (0, H - 1):
            add_col(g[r, c], c)

    out = g.copy()

    def components_8(mask):
        """Return 8-connected components of True pixels in mask."""
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        comps = []
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not seen[r, c]:
                    stack = [(r, c)]
                    seen[r, c] = True
                    cells = []
                    while stack:
                        rr, cc = stack.pop()
                        cells.append((rr, cc))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = rr + dr, cc + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    if mask[nr, nc] and not seen[nr, nc]:
                                        seen[nr, nc] = True
                                        stack.append((nr, nc))
                    comps.append(cells)
        return comps

    def in_interior(r, c):
        return 1 <= r <= H - 2 and inner_left <= c <= inner_right

    placements = []
    left_region = g[:, :sep_col]
    move_colors = set(row_marks.keys()) & set(col_marks.keys())

    # ------------------------------------------------------------------
    # For each valid color, erase left-margin source component(s)
    # and create placements at all valid interior intersections.
    # ------------------------------------------------------------------
    for color in move_colors:
        color = int(color)

        targets = []
        for r in row_marks.get(color, ()):
            if 1 <= r <= H - 2:
                for c in col_marks.get(color, ()):
                    if inner_left <= c <= inner_right:
                        targets.append((r, c))

        if not targets or left_region.size == 0:
            continue

        mask = (left_region == color)
        if not np.any(mask):
            continue

        for cells in components_8(mask):
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]

            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs), max(cs)

            center_r = (min_r + max_r) // 2
            center_c = (min_c + max_c) // 2

            rel_cells = [(r - center_r, c - center_c) for r, c in cells]

            # Erase the source component.
            for r, c in cells:
                out[r, c] = 0

            # Create one placement for each target center.
            for tr, tc in targets:
                placements.append((tr, tc, color, rel_cells))

    # Draw from bottom to top so that upper placements win overlaps.
    placements.sort(key=lambda p: (p[0], p[1]), reverse=True)

    for tr, tc, color, rel_cells in placements:
        for dr, dc in rel_cells:
            rr, cc = tr + dr, tc + dc
            if in_interior(rr, cc):
                out[rr, cc] = color

    return out