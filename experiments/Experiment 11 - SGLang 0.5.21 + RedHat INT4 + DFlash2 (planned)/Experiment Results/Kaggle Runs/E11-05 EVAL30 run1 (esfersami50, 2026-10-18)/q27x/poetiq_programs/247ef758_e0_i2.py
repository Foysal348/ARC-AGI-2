import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC grid by stamping left-side patterns onto a right-side frame.

    Rule:
      - Find the vertical separator column.
      - The region immediately to its right forms a rectangular frame.
      - The dominant non-zero color on that frame border is the frame background.
      - Non-background colored cells on the frame edges are markers:
          * cells on the left/right vertical frame edges define row positions
          * cells on the top/bottom horizontal frame edges define column positions
      - Each colored shape in the left region is a stamp.
      - If a color has a left-side shape and at least one row marker and one
        column marker, remove that shape from the left region and stamp it at
        every (row marker, column marker) intersection.
      - Stamps are centered at the center of the shape's bounding box.
      - Unused left-side shapes are preserved.
      - Overlapping stamps keep the smaller color value.
    """
    g = np.array(grid, dtype=int, copy=True)

    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g

    # -------------------------------------------------------------------------
    # 1. Find the vertical separator column.
    #
    # It is usually a full-height constant non-zero column with content on both
    # sides. If several candidates exist, prefer the one whose assumed frame
    # border has a strong dominant non-zero color.
    # -------------------------------------------------------------------------
    candidates = []
    for c in range(1, W - 1):
        col = g[:, c]
        if col.size == 0:
            continue

        val = int(col[0])
        if val != 0 and np.all(col == val):
            if np.count_nonzero(g[:, :c]) > 0 and np.count_nonzero(g[:, c + 1:]) > 0:
                candidates.append(c)

    if candidates:
        def candidate_score(c: int) -> float:
            left_border = c + 1
            if left_border >= W:
                return -1.0

            mask = np.zeros((H, W), dtype=bool)
            mask[0, left_border:] = True
            mask[-1, left_border:] = True

            if H > 2:
                mask[1:-1, left_border] = True
                mask[1:-1, W - 1] = True

            border = g[mask]
            if border.size == 0:
                return -1.0

            vals, counts = np.unique(border, return_counts=True)
            non_zero = vals != 0

            if not np.any(non_zero):
                return 0.0

            # A real frame usually has one dominant border color.
            return float(counts[non_zero].max()) / float(border.size)

        best_c = candidates[0]
        best_score = candidate_score(best_c)

        for c in candidates[1:]:
            score = candidate_score(c)
            if score > best_score + 1e-9 or (
                abs(score - best_score) <= 1e-9 and c < best_c
            ):
                best_c = c
                best_score = score

        sep = best_c

    else:
        # Fallback: any full constant non-zero interior column.
        candidates = [
            c for c in range(1, W - 1)
            if g[:, c].size > 0 and int(g[0, c]) != 0 and np.all(g[:, c] == g[0, c])
        ]
        if not candidates:
            return g

        sep = min(candidates)

    left_border = sep + 1
    if left_border >= W:
        return g

    # -------------------------------------------------------------------------
    # 2. Determine the frame background color.
    #    Count border cells; if zeros dominate, background is 0.
    # -------------------------------------------------------------------------
    frame_mask = np.zeros((H, W), dtype=bool)
    frame_mask[0, left_border:] = True
    frame_mask[-1, left_border:] = True

    if H > 2:
        frame_mask[1:-1, left_border] = True
        frame_mask[1:-1, W - 1] = True

    border_vals = g[frame_mask]
    vals, counts = np.unique(border_vals, return_counts=True)
    non_zero = vals != 0

    if np.any(non_zero):
        non_zero_vals = vals[non_zero]
        non_zero_counts = counts[non_zero]
        max_non_zero = int(non_zero_counts.max())
        zero_count = int(np.sum(counts[~non_zero])) if np.any(~non_zero) else 0

        if zero_count > max_non_zero:
            bg_color = 0
        else:
            bg_color = int(non_zero_vals[non_zero_counts.argmax()])
    else:
        bg_color = 0

    # -------------------------------------------------------------------------
    # 3. Collect row and column markers on the frame border.
    #    Corners are ignored to avoid confusing border corners with markers.
    # -------------------------------------------------------------------------
    row_marks = {}
    col_marks = {}

    # Vertical markers: left and right frame borders, excluding top/bottom rows.
    for r in range(1, H - 1):
        for c in (left_border, W - 1):
            v = int(g[r, c])
            if v != 0 and v != bg_color:
                row_marks.setdefault(v, set()).add(r)

    # Horizontal markers: top and bottom frame borders, excluding side corners.
    for c in range(left_border + 1, W - 1):
        for r in (0, H - 1):
            v = int(g[r, c])
            if v != 0 and v != bg_color:
                col_marks.setdefault(v, set()).add(c)

    # -------------------------------------------------------------------------
    # 4. Extract left-side shapes.
    # -------------------------------------------------------------------------
    left = g[:, :sep]

    if left.size == 0:
        left_colors = set()
    else:
        left_colors = {int(x) for x in np.unique(left) if int(x) != 0}

    def extract_shape(color: int):
        mask = (left == color)
        if not np.any(mask):
            return None

        ys, xs = np.where(mask)

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2

        relative_cells = sorted(
            {(int(y) - cy, int(x) - cx) for y, x in zip(ys, xs)}
        )
        return relative_cells if relative_cells else None

    shapes = {}
    for color in left_colors:
        shape = extract_shape(color)
        if shape is not None:
            shapes[color] = shape

    # A color is stamped only if it has both a left shape and frame markers.
    used_colors = {
        color
        for color in left_colors
        if shapes.get(color) and row_marks.get(color) and col_marks.get(color)
    }

    # -------------------------------------------------------------------------
    # 5. Build output:
    #    - keep separator
    #    - keep frame border unchanged
    #    - keep unused left-side shapes
    #    - zero the rest initially
    # -------------------------------------------------------------------------
    out = np.zeros_like(g)

    out[:, sep] = g[:, sep]

    # Copy the frame border.
    out[0, left_border:] = g[0, left_border:]
    out[-1, left_border:] = g[-1, left_border:]

    if H > 2:
        out[1:-1, left_border] = g[1:-1, left_border]
        out[1:-1, W - 1] = g[1:-1, W - 1]

    # Preserve unused left-side shapes.
    if sep > 0:
        retained_colors = left_colors - used_colors
        if retained_colors:
            retain_mask = np.zeros_like(left, dtype=bool)
            for color in retained_colors:
                retain_mask |= (left == color)
            out[:, :sep] = np.where(retain_mask, left, 0)

    # -------------------------------------------------------------------------
    # 6. Stamp shapes at marker intersections.
    #    Lower colors are processed first and do not overwrite existing cells,
    #    so smaller color IDs have priority in overlaps.
    # -------------------------------------------------------------------------
    if H > 2 and left_border + 1 <= W - 2:
        row_min, row_max = 1, H - 2
        col_min, col_max = left_border + 1, W - 2

        for color in sorted(used_colors):
            shape = shapes.get(color)
            if not shape:
                continue

            for r in sorted(row_marks[color]):
                for c in sorted(col_marks[color]):
                    for dr, dc in shape:
                        nr = r + dr
                        nc = c + dc

                        if (
                            row_min <= nr <= row_max
                            and col_min <= nc <= col_max
                            and out[nr, nc] == 0
                        ):
                            out[nr, nc] = color

    return out