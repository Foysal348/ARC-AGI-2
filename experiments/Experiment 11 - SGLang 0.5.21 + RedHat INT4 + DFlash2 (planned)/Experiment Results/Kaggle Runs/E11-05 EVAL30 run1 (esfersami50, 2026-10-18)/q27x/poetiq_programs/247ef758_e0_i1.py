import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the ARC grid by:
      1. Finding the vertical separator column.
      2. Keeping the rectangular frame to the right of the separator unchanged.
      3. Reading small colored "stamps" from the left side.
      4. For each stamp color that has matching markers on the frame, stamp copies
         at all intersections of that color's row markers and column markers.
      5. If stamps overlap, lower color values take precedence, as implied by the
         examples where 4 remains visible over 7.
    """
    g = np.array(grid, dtype=int, copy=True)
    if g.size == 0:
        return g

    H, W = g.shape

    # ------------------------------------------------------------------
    # Find the vertical separator column.
    # It is a full-height constant non-zero column. We prefer the rightmost
    # such column that is not the final column and has content on both sides.
    # ------------------------------------------------------------------
    candidates = []
    for c in range(1, W - 1):
        col = g[:, c]
        if col.size > 0 and np.all(col == col[0]) and col[0] != 0:
            if np.count_nonzero(g[:, :c]) > 0 and np.count_nonzero(g[:, c + 1:]) > 0:
                candidates.append(c)

    if not candidates:
        # Fallback: any full constant non-zero column before the last column.
        candidates = [
            c for c in range(1, W - 1)
            if g[:, c].size > 0 and np.all(g[:, c] == g[:, c, 0]) and g[0, c] != 0
        ]

    if not candidates:
        return g

    sep = max(candidates)

    # If the chosen separator leaves no room for the frame/interior, return safely.
    if sep + 2 >= W or H < 3:
        return g

    left_border = sep + 1
    right_border = W - 1

    # ------------------------------------------------------------------
    # Identify the dominant non-zero frame background color.
    # This color fills the frame and is not treated as a marker.
    # ------------------------------------------------------------------
    border = np.zeros_like(g, dtype=bool)
    border[0, left_border:] = True
    border[-1, left_border:] = True
    border[1:-1, left_border] = True
    border[1:-1, right_border] = True

    border_vals = g[border]
    border_vals = border_vals[border_vals != 0]
    bg_color = None
    if border_vals.size > 0:
        vals, counts = np.unique(border_vals, return_counts=True)
        max_count = counts.max()
        # Use it only if it is reasonably dominant as a background.
        if max_count >= max(3, len(border_vals) // 2):
            bg_color = int(vals[counts.argmax()])

    # ------------------------------------------------------------------
    # Collect frame markers:
    #   row markers: on the left/right vertical frame borders, excluding corners
    #   col markers: on the top/bottom horizontal frame borders, excluding corners
    # ------------------------------------------------------------------
    row_marks = defaultdict(set)
    col_marks = defaultdict(set)

    # Vertical markers
    if H > 2:
        marker_cols = {left_border, right_border}
        for r in range(1, H - 1):
            for c in marker_cols:
                if 0 <= c < W:
                    v = int(g[r, c])
                    if v != 0 and v != bg_color:
                        row_marks[v].add(r)

    # Horizontal markers
    if W > sep + 2:
        marker_rows = {0, H - 1}
        for c in range(left_border + 1, W - 1):
            for r in marker_rows:
                if 0 <= r < H:
                    v = int(g[r, c])
                    if v != 0 and v != bg_color:
                        col_marks[v].add(c)

    # ------------------------------------------------------------------
    # Determine colors on the left side.
    # Colors that have both row and column markers are stamped.
    # Other left colors are preserved in their original left-side positions.
    # ------------------------------------------------------------------
    left_region = g[:, :sep]
    left_colors = set(left_region[left_region != 0].tolist())
    left_colors.discard(int(g[0, sep]))       # separator color is not a stamp
    if bg_color is not None:
        left_colors.discard(int(bg_color))    # frame background is not a stamp

    used_colors = set()
    for c in left_colors:
        if row_marks.get(c) and col_marks.get(c):
            used_colors.add(c)

    retained_colors = left_colors - used_colors

    # ------------------------------------------------------------------
    # Initialize output: zeros, then copy separator and frame borders.
    # ------------------------------------------------------------------
    out = np.zeros_like(g)

    # Separator column
    out[:, sep] = g[:, sep]

    # Frame borders
    out[0, left_border:] = g[0, left_border:]
    out[-1, left_border:] = g[-1, left_border:]
    if H > 1:
        out[1:-1, left_border] = g[1:-1, left_border]
        out[1:-1, right_border] = g[1:-1, right_border]

    # Preserve retained left-side colors
    retain_mask = np.zeros_like(g, dtype=bool)
    for c in retained_colors:
        retain_mask |= (left_region == c)
    if np.any(retain_mask):
        out[:, :sep] = np.where(retain_mask[:, :sep], left_region, 0)

    # ------------------------------------------------------------------
    # Extract relative stamp shapes from the left side.
    # The anchor is the center of the bounding box of that color's left cells.
    # ------------------------------------------------------------------
    shapes = {}
    for c in used_colors:
        mask = (left_region == c)
        if not np.any(mask):
            continue
        ys, xs = np.where(mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        cy = (int(y_min) + int(y_max)) // 2
        cx = (int(x_min) + int(x_max)) // 2
        rel = [(int(y) - cy, int(x) - cx) for y, x in zip(ys, xs)]
        rel = sorted(set(rel))
        if rel:
            shapes[c] = rel

    # ------------------------------------------------------------------
    # Stamp used colors at all marker intersections.
    # Lower colors are processed first and existing non-zero cells are not
    # overwritten, so lower color values have precedence in overlaps.
    # ------------------------------------------------------------------
    interior_r = range(1, H - 1)
    interior_c = range(left_border + 1, W - 1)

    for c in sorted(used_colors):
        if c not in shapes:
            continue
        for r in sorted(row_marks[c]):
            for cc in sorted(col_marks[c]):
                for dy, dx in shapes[c]:
                    nr = r + dy
                    nc = cc + dx
                    if 1 <= nr < H - 1 and left_border + 1 <= nc < W - 1:
                        if out[nr, nc] == 0:
                            out[nr, nc] = c

    return out