import numpy as np
from collections import Counter, defaultdict


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input by using the left-side legend to stamp objects into the
    right-side framed canvas.

    Details:
    - Find the vertical divider: a full-height column of one non-zero color.
    - The right of the divider contains a rectangular frame.
    - Determine the frame's base color as the most frequent non-zero color on its
      perimeter.
    - Marker colors are non-base colors on the frame perimeter.
        * A valid vertical marker column for a color must appear at the same column
          on both the top and bottom borders.
        * A valid horizontal marker row must appear at the same row on both the
          left and right borders.
    - Group non-zero cells left of the divider by color. Each color is a template.
    - If a template color has both valid rows and valid columns, remove the template
      from the left side and stamp it at all intersections of those rows/columns.
    - Each template is anchored at the center of its bounding box.
    - Templates are processed in top-to-bottom legend order, so later templates
      overwrite earlier overlapping pixels.
    """
    g = np.array(grid, dtype=int, copy=True)

    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g

    # ------------------------------------------------------------------
    # 1. Find the vertical separator column.
    # ------------------------------------------------------------------
    sep = None
    for c in range(W):
        if np.all(g[:, c] == g[0, c]) and g[0, c] != 0:
            sep = c
            break

    # Fallback: choose the most uniform non-zero column.
    if sep is None:
        best_c, best_count = None, -1
        for c in range(W):
            vals, counts = np.unique(g[:, c], return_counts=True)
            for val, cnt in zip(vals, counts):
                if val != 0 and int(cnt) > best_count:
                    best_count = int(cnt)
                    best_c = c
        if best_c is None or best_c >= W - 1:
            return g
        sep = best_c

    if sep >= W - 1:
        return g

    # ------------------------------------------------------------------
    # 2. Find the frame bounding box on the right side.
    # ------------------------------------------------------------------
    right = g[:, sep + 1:]
    if right.size == 0 or not np.any(right):
        return g

    row_has = np.any(right != 0, axis=1)
    col_has = np.any(right != 0, axis=0)

    if not np.any(row_has) or not np.any(col_has):
        return g

    top = int(np.argmax(row_has))
    bottom = int(H - 1 - np.argmax(row_has[::-1]))

    col_indices = np.where(col_has)[0]
    left = sep + 1 + int(col_indices[0])
    right_edge = sep + 1 + int(col_indices[-1])

    if top >= bottom or left >= right_edge:
        return g

    # ------------------------------------------------------------------
    # 3. Determine the frame's base border color.
    # ------------------------------------------------------------------
    perimeter_values = []

    for c in range(left, right_edge + 1):
        perimeter_values.append(int(g[top, c]))
        perimeter_values.append(int(g[bottom, c]))

    for r in range(top + 1, bottom):
        perimeter_values.append(int(g[r, left]))
        perimeter_values.append(int(g[r, right_edge]))

    counter = Counter(v for v in perimeter_values if v != 0)
    if not counter:
        return g

    max_count = max(counter.values())
    base_candidates = [color for color, cnt in counter.items() if cnt == max_count]

    # Prefer the bottom-right corner if it is among the most common colors.
    corner_color = int(g[bottom, right_edge])
    if corner_color in base_candidates:
        base = corner_color
    else:
        base = base_candidates[0]

    # ------------------------------------------------------------------
    # 4. Collect valid marker rows and columns by color.
    # ------------------------------------------------------------------
    top_cols = defaultdict(set)
    bottom_cols = defaultdict(set)

    for c in range(left, right_edge + 1):
        v_top = int(g[top, c])
        if v_top != 0 and v_top != base:
            top_cols[v_top].add(c)

        v_bottom = int(g[bottom, c])
        if v_bottom != 0 and v_bottom != base:
            bottom_cols[v_bottom].add(c)

    left_rows = defaultdict(set)
    right_rows = defaultdict(set)

    for r in range(top, bottom + 1):
        v_left = int(g[r, left])
        if v_left != 0 and v_left != base:
            left_rows[v_left].add(r)

        v_right = int(g[r, right_edge])
        if v_right != 0 and v_right != base:
            right_rows[v_right].add(r)

    marker_colors = (
        set(top_cols.keys())
        | set(bottom_cols.keys())
        | set(left_rows.keys())
        | set(right_rows.keys())
    )

    valid_rows = {}
    valid_cols = {}

    for color in marker_colors:
        vr = left_rows.get(color, set()) & right_rows.get(color, set())
        vc = top_cols.get(color, set()) & bottom_cols.get(color, set())

        if vr:
            valid_rows[color] = vr
        if vc:
            valid_cols[color] = vc

    # ------------------------------------------------------------------
    # 5. Extract legend templates from the left side.
    # ------------------------------------------------------------------
    legend = defaultdict(list)

    for r in range(H):
        for c in range(sep):
            v = int(g[r, c])
            if v != 0:
                legend[v].append((r, c))

    used_colors = [
        color for color in legend
        if color in valid_rows and color in valid_cols
    ]

    if not used_colors:
        return g

    # Remove used legend objects from the left side.
    for color in used_colors:
        for r, c in legend[color]:
            g[r, c] = 0

    # ------------------------------------------------------------------
    # 6. Stamp used templates at all valid intersections.
    # ------------------------------------------------------------------
    def legend_order_key(color):
        coords = legend[color]
        r0 = min(r for r, _ in coords)
        c0 = min(c for _, c in coords)
        return (r0, c0, color)

    for color in sorted(used_colors, key=legend_order_key):
        coords = legend[color]

        rmin = min(r for r, _ in coords)
        rmax = max(r for r, _ in coords)
        cmin = min(c for _, c in coords)
        cmax = max(c for _, c in coords)

        anchor_r = (rmin + rmax) // 2
        anchor_c = (cmin + cmax) // 2

        offsets = [(r - anchor_r, c - anchor_c) for r, c in coords]

        for cr in sorted(valid_rows[color]):
            for cc in sorted(valid_cols[color]):
                for dr, dc in offsets:
                    rr = cr + dr
                    ccc = cc + dc

                    if 0 <= rr < H and 0 <= ccc < W:
                        # Draw only inside the frame interior.
                        if top < rr < bottom and left < ccc < right_edge:
                            g[rr, ccc] = color

    return g