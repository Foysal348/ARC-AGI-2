import numpy as np
from collections import defaultdict, Counter


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Stamp legend patterns into a framed canvas.

    The input contains:
      - a vertical divider column that is fully filled with one non-zero color,
      - a legend of colored patterns to the left of the divider,
      - a rectangular frame to the right of the divider.

    Non-base colored cells on the frame perimeter are markers:
      - top/bottom markers provide column coordinates,
      - left/right markers provide row coordinates.

    For every legend color that has at least one row marker and one column marker,
    the legend pattern is removed from the left side and stamped inside the frame
    at all intersections of its marker rows and marker columns.

    Overlapping stamps are resolved by processing centers in row-major order
    (top to bottom, left to right) and only writing into cells that are still
    empty or the frame base color. The frame border is preserved.
    """
    g = np.array(grid, dtype=int, copy=True)

    if g.ndim != 2:
        return g

    H, W = g.shape
    if H < 3 or W < 3:
        return g

    # ------------------------------------------------------------------
    # 1. Find the vertical divider.
    # ------------------------------------------------------------------
    divider_candidates = []
    for c in range(W - 1):
        v = int(g[0, c])
        if v == 0:
            continue

        if not np.all(g[:, c] == v):
            continue

        # The frame should begin immediately to the right.
        if not np.any(g[:, c + 1] != 0):
            continue

        right = g[:, c + 1:]
        row_idx = np.where(np.any(right != 0, axis=1))[0]
        col_idx = np.where(np.any(right != 0, axis=0))[0]

        if row_idx.size < 2 or col_idx.size < 2:
            continue

        top = int(row_idx[0])
        bottom = int(row_idx[-1])
        left = c + 1 + int(col_idx[0])
        right_edge = c + 1 + int(col_idx[-1])

        if top >= bottom or left >= right_edge:
            continue

        divider_candidates.append(c)

    if not divider_candidates:
        return g

    # Use the leftmost valid divider.
    divider = divider_candidates[0]

    # ------------------------------------------------------------------
    # 2. Find the frame bounding box on the right side.
    # ------------------------------------------------------------------
    right = g[:, divider + 1:]
    row_idx = np.where(np.any(right != 0, axis=1))[0]
    col_idx = np.where(np.any(right != 0, axis=0))[0]

    if row_idx.size < 2 or col_idx.size < 2:
        return g

    top = int(row_idx[0])
    bottom = int(row_idx[-1])
    left = divider + 1 + int(col_idx[0])
    right_edge = divider + 1 + int(col_idx[-1])

    if top >= bottom or left >= right_edge:
        return g

    # ------------------------------------------------------------------
    # 3. Determine the frame's base border color.
    # ------------------------------------------------------------------
    border_values = []

    for c in range(left, right_edge + 1):
        border_values.append(int(g[top, c]))
        border_values.append(int(g[bottom, c]))

    for r in range(top + 1, bottom):
        border_values.append(int(g[r, left]))
        border_values.append(int(g[r, right_edge]))

    counter = Counter(v for v in border_values if v != 0)
    if not counter:
        return g

    corners = [
        int(g[top, left]),
        int(g[top, right_edge]),
        int(g[bottom, left]),
        int(g[bottom, right_edge]),
    ]
    nonzero_corners = [x for x in corners if x != 0]

    if len(set(nonzero_corners)) == 1:
        base = nonzero_corners[0]
    else:
        max_count = max(counter.values())
        base_candidates = [color for color, count in counter.items() if count == max_count]

        if len(base_candidates) == 1:
            base = base_candidates[0]
        else:
            def base_score(color):
                corner_count = sum(1 for x in corners if x == color)
                return (corner_count, -color)

            base = max(base_candidates, key=base_score)

    # ------------------------------------------------------------------
    # 4. Collect marker rows and columns.
    # ------------------------------------------------------------------
    row_markers = defaultdict(set)
    col_markers = defaultdict(set)
    border_cells = defaultdict(set)

    # Top and bottom borders provide column markers.
    for c in range(left, right_edge + 1):
        for r in (top, bottom):
            v = int(g[r, c])
            if v != 0 and v != base:
                col_markers[v].add(c)
                border_cells[v].add((r, c))

    # Left and right borders provide row markers.
    for r in range(top, bottom + 1):
        for c in (left, right_edge):
            v = int(g[r, c])
            if v != 0 and v != base:
                row_markers[v].add(r)
                border_cells[v].add((r, c))

    # ------------------------------------------------------------------
    # 5. Extract legend patterns from the left side.
    # ------------------------------------------------------------------
    legend = defaultdict(list)
    for r in range(H):
        for c in range(divider):
            v = int(g[r, c])
            if v != 0:
                legend[v].append((r, c))

    active_colors = []
    for color in legend:
        if (
            row_markers.get(color)
            and col_markers.get(color)
            and len(border_cells.get(color, ())) >= 2
        ):
            active_colors.append(color)

    if not active_colors:
        return g

    # Erase active legend patterns.
    for color in active_colors:
        for r, c in legend[color]:
            g[r, c] = 0

    # ------------------------------------------------------------------
    # 6. Build templates anchored at bounding-box centers.
    # ------------------------------------------------------------------
    templates = {}
    for color in active_colors:
        coords = legend[color]
        if not coords:
            continue

        rmin = min(r for r, _ in coords)
        rmax = max(r for r, _ in coords)
        cmin = min(c for _, c in coords)
        cmax = max(c for _, c in coords)

        anchor_r = (rmin + rmax) // 2
        anchor_c = (cmin + cmax) // 2

        offsets = set()
        for r, c in coords:
            offsets.add((r - anchor_r, c - anchor_c))

        templates[color] = list(offsets)

    # ------------------------------------------------------------------
    # 7. Create stamp placements.
    # ------------------------------------------------------------------
    placements = []
    for color in active_colors:
        offsets = templates[color]

        for cr in row_markers[color]:
            for cc in col_markers[color]:
                inside = set()

                for dr, dc in offsets:
                    rr = cr + dr
                    c2 = cc + dc

                    # Only stamp strictly inside the frame.
                    if top < rr < bottom and left < c2 < right_edge:
                        inside.add((rr, c2))

                if inside:
                    # Priority: topmost center, then leftmost center, then color.
                    placements.append(((cr, cc, color), color, inside))

    placements.sort(key=lambda item: item[0])

    # ------------------------------------------------------------------
    # 8. Stamp patterns without overwriting previously placed pixels.
    # ------------------------------------------------------------------
    for _, color, inside in placements:
        for rr, cc in inside:
            if g[rr, cc] == 0 or g[rr, cc] == base:
                g[rr, cc] = color

    return g