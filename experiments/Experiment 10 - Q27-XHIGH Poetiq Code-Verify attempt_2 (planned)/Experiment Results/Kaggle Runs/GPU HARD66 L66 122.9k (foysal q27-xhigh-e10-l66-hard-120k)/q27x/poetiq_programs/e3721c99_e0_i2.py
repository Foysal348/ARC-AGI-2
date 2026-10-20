import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recolor target components (normally color 5) according to their number of holes.

    The input contains an ordered palette of small colored components.
    A separator made of color 1 is ignored. The palette is ordered spatially:
    left-to-right for horizontal arrangements, top-to-bottom for vertical ones.

    Each target component is given the palette color whose position equals the
    number of enclosed holes in the target component. If a target component has
    more holes than there are palette colors, it is erased to 0.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2:
        return grid.copy()

    h, w = grid.shape
    if h == 0 or w == 0:
        return grid.copy()

    out = grid.copy()
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # ------------------------------------------------------------------
    # Label all non-zero 4-connected components.
    # ------------------------------------------------------------------
    labels = np.zeros((h, w), dtype=int)

    comp_color = [0]
    comp_top = [(0, 0)]
    comp_border = [0]
    comp_size = [0]

    next_id = 0

    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or labels[r, c] != 0:
                continue

            next_id += 1
            color = int(grid[r, c])

            labels[r, c] = next_id
            stack = [(r, c)]

            size = 0
            min_r = h
            min_c = w
            sides = 0

            while stack:
                cr, cc = stack.pop()
                size += 1

                if cr < min_r:
                    min_r = cr
                    min_c = cc
                elif cr == min_r and cc < min_c:
                    min_c = cc

                if cr == 0:
                    sides |= 1
                if cr == h - 1:
                    sides |= 2
                if cc == 0:
                    sides |= 4
                if cc == w - 1:
                    sides |= 8

                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if grid[nr, nc] == color and labels[nr, nc] == 0:
                            labels[nr, nc] = next_id
                            stack.append((nr, nc))

            comp_color.append(color)
            comp_top.append((min_r, min_c))
            comp_border.append(sides)
            comp_size.append(size)

    def side_count(mask: int) -> int:
        return bin(mask).count("1")

    # ------------------------------------------------------------------
    # Separator components are color 1 components touching at least 2 borders.
    # ------------------------------------------------------------------
    separator_ids = set()
    for cid in range(1, next_id + 1):
        if comp_color[cid] == 1 and side_count(comp_border[cid]) >= 2:
            separator_ids.add(cid)

    # ------------------------------------------------------------------
    # Target color is normally 5. If 5 is absent, use the largest non-separator color.
    # ------------------------------------------------------------------
    if np.any(grid == 5):
        target_color = 5
    else:
        color_counts = {}
        for cid in range(1, next_id + 1):
            if cid in separator_ids:
                continue
            color = comp_color[cid]
            color_counts[color] = color_counts.get(color, 0) + comp_size[cid]

        if not color_counts:
            return out

        target_color = max(color_counts, key=lambda x: color_counts[x])

    target_ids = [
        cid
        for cid in range(1, next_id + 1)
        if comp_color[cid] == target_color and cid not in separator_ids
    ]

    if not target_ids:
        return out

    # ------------------------------------------------------------------
    # Palette components are all non-target, non-separator components.
    # ------------------------------------------------------------------
    palette_ids = [
        cid
        for cid in range(1, next_id + 1)
        if comp_color[cid] != target_color and cid not in separator_ids
    ]

    if not palette_ids:
        return out

    # Order palette spatially.
    if len(palette_ids) > 1:
        rows = [comp_top[cid][0] for cid in palette_ids]
        cols = [comp_top[cid][1] for cid in palette_ids]

        row_span = max(rows) - min(rows)
        col_span = max(cols) - min(cols)

        if col_span > row_span:
            palette_ids.sort(key=lambda cid: (comp_top[cid][1], comp_top[cid][0]))
        elif row_span > col_span:
            palette_ids.sort(key=lambda cid: (comp_top[cid][0], comp_top[cid][1]))
        else:
            palette_ids.sort(key=lambda cid: (comp_top[cid][0], comp_top[cid][1]))
    else:
        palette_ids.sort(key=lambda cid: (comp_top[cid][0], comp_top[cid][1]))

    palette_colors = [comp_color[cid] for cid in palette_ids]

    # ------------------------------------------------------------------
    # Count enclosed holes for each target component.
    # A hole is a 4-connected zero region that does not touch the grid border
    # and whose neighboring non-zero cells all belong to exactly one component.
    # ------------------------------------------------------------------
    holes = {cid: 0 for cid in target_ids}

    zero_labels = np.zeros((h, w), dtype=int)
    next_zero = 0

    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 or zero_labels[r, c] != 0:
                continue

            next_zero += 1
            zero_labels[r, c] = next_zero
            stack = [(r, c)]

            touches_border = False
            cells = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))

                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True

                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if grid[nr, nc] == 0 and zero_labels[nr, nc] == 0:
                            zero_labels[nr, nc] = next_zero
                            stack.append((nr, nc))

            if touches_border:
                continue

            adjacent_components = set()

            for cr, cc in cells:
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if grid[nr, nc] != 0:
                            cid = int(labels[nr, nc])
                            if cid != 0:
                                adjacent_components.add(cid)

            if len(adjacent_components) == 1:
                cid = next(iter(adjacent_components))
                if cid in holes:
                    holes[cid] += 1

    # ------------------------------------------------------------------
    # Apply mapping.
    # ------------------------------------------------------------------
    for cid in target_ids:
        hole_count = holes[cid]

        if hole_count < len(palette_colors):
            out[labels == cid] = palette_colors[hole_count]
        else:
            out[labels == cid] = 0

    return out