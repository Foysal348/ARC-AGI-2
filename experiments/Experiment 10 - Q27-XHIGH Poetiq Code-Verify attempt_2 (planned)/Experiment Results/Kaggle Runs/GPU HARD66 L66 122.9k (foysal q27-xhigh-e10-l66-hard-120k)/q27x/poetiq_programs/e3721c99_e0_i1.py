import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each connected component of the target color (normally 5)
    according to the number of enclosed holes it has.

    Reference colors are learned from non-target components:
    - A reference component encodes a color for a given number of holes.
    - Holes are 4-connected zero regions not touching the grid border.
    - A zero region is counted as a hole only if its neighboring non-zero
      cells are all the same color.
    - 1-colored separator components touching at least two borders are ignored.
    - If multiple references have the same hole count, prefer non-border,
      then smaller size, then top-left position.
    """
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    if grid.size == 0:
        return out

    h, w = grid.shape
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # Count colors and choose the target color.
    # In all given cases it is 5; if 5 is absent, use the most common color.
    unique, counts = np.unique(grid, return_counts=True)
    color_count = {}
    for v, c in zip(unique, counts):
        v, c = int(v), int(c)
        if v != 0:
            color_count[v] = c

    if not color_count:
        return out

    if 5 in color_count:
        target_color = 5
    else:
        target_color = max(color_count, key=lambda c: color_count[c])

    nonzero_colors = set(color_count.keys())

    def label_components(mask: np.ndarray):
        """Label 4-connected components of a boolean mask."""
        labels = np.zeros((h, w), dtype=int)
        next_id = 0

        for r in range(h):
            for c in range(w):
                if mask[r, c] and labels[r, c] == 0:
                    next_id += 1
                    stack = [(r, c)]
                    labels[r, c] = next_id

                    while stack:
                        cr, cc = stack.pop()

                        nr = cr + 1
                        if nr < h and mask[nr, cc] and labels[nr, cc] == 0:
                            labels[nr, cc] = next_id
                            stack.append((nr, cc))

                        nr = cr - 1
                        if nr >= 0 and mask[nr, cc] and labels[nr, cc] == 0:
                            labels[nr, cc] = next_id
                            stack.append((nr, cc))

                        nc = cc + 1
                        if nc < w and mask[cr, nc] and labels[cr, nc] == 0:
                            labels[cr, nc] = next_id
                            stack.append((cr, nc))

                        nc = cc - 1
                        if nc >= 0 and mask[cr, nc] and labels[cr, nc] == 0:
                            labels[cr, nc] = next_id
                            stack.append((cr, nc))

        return labels, next_id

    # Label all non-zero components by color.
    labels = np.zeros((h, w), dtype=int)
    next_id = 0

    comp_color = {}
    comp_size = {}
    comp_top = {}
    comp_border_sides = {}

    for color in sorted(nonzero_colors):
        mask = (grid == color)
        local_labels, n = label_components(mask)

        for lid in range(1, n + 1):
            next_id += 1
            cells = (local_labels == lid)
            labels[cells] = next_id

            ys, xs = np.nonzero(cells)
            comp_color[next_id] = color
            comp_size[next_id] = int(ys.size)

            min_r = int(ys.min())
            min_c = int(xs[ys == min_r].min())
            comp_top[next_id] = (min_r, min_c)

            sides = 0
            if np.any(ys == 0):
                sides += 1
            if np.any(ys == h - 1):
                sides += 1
            if np.any(xs == 0):
                sides += 1
            if np.any(xs == w - 1):
                sides += 1
            comp_border_sides[next_id] = sides

    # Label zero regions and identify which touch the border.
    zero_labels, n_zero = label_components(grid == 0)
    zero_border = [False] * (n_zero + 1)

    for zid in range(1, n_zero + 1):
        ys, xs = np.nonzero(zero_labels == zid)
        if (
            np.any(ys == 0)
            or np.any(ys == h - 1)
            or np.any(xs == 0)
            or np.any(xs == w - 1)
        ):
            zero_border[zid] = True

    # Count holes for each non-zero component.
    holes = {}

    for zid in range(1, n_zero + 1):
        if zero_border[zid]:
            continue

        ys, xs = np.nonzero(zero_labels == zid)
        adjacent_colors = set()
        adjacent_ids = set()

        for r, c in zip(ys, xs):
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    val = int(grid[nr, nc])
                    if val != 0:
                        adjacent_colors.add(val)
                        cid = int(labels[nr, nc])
                        if cid != 0:
                            adjacent_ids.add(cid)

        # Count as a hole only if the empty region is surrounded by one color.
        if len(adjacent_colors) == 1 and adjacent_ids:
            for cid in adjacent_ids:
                holes[cid] = holes.get(cid, 0) + 1

    # Build mapping: hole count -> reference color.
    candidates = {}

    for cid in range(1, next_id + 1):
        if comp_color[cid] == target_color:
            continue

        # Ignore 1-colored divider components that touch at least two borders.
        if comp_color[cid] == 1 and comp_border_sides[cid] >= 2:
            continue

        hole_count = holes.get(cid, 0)
        priority = (
            comp_border_sides[cid] > 0,  # prefer non-border references
            comp_size[cid],              # prefer smaller shapes
            comp_top[cid],               # top-left tie-breaker
            cid                          # final deterministic tie-breaker
        )
        candidates.setdefault(hole_count, []).append((priority, cid))

    hole_to_color = {}
    for hole_count, cands in candidates.items():
        _, best_cid = min(cands, key=lambda x: x[0])
        hole_to_color[hole_count] = comp_color[best_cid]

    # Recolor target components.
    for cid in range(1, next_id + 1):
        if comp_color[cid] == target_color:
            hole_count = holes.get(cid, 0)
            if hole_count in hole_to_color:
                out[labels == cid] = hole_to_color[hole_count]

    return out