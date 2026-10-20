import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    ARC transformation:

    - The background is the most frequent color.
    - The largest 4-connected non-background component is the template object.
    - All other non-background pixels form a sparse dot pattern.
    - The pattern center is the dot closest to the pattern centroid,
      with preference given to a dot having the template color.
    - Each dot is snapped to a 2-spaced even lattice. For every dot,
      a copy of the template is placed at:
          original_template_top_left + (dy / 2) * template_height,
                                     (dx / 2) * template_width
      where (dx, dy) are snapped relative coordinates.
    - The center dot's copy is colored with the pattern color
      (most frequent non-template color among pattern dots);
      all other copies use the template color.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # Background color is the most common color.
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    nonbg = grid != bg
    if not nonbg.any():
        return grid.copy()

    # Find all 4-connected non-background components.
    visited = np.zeros((h, w), dtype=bool)
    best_cells = []
    best_size = -1
    best_area = -1

    for r in range(h):
        for c in range(w):
            if not nonbg[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            cells = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if nonbg[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

            # Prefer larger pixel count, then larger bounding-box area.
            if len(cells) > best_size:
                rows = [x[0] for x in cells]
                cols = [x[1] for x in cells]
                area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
                best_cells = cells
                best_size = len(cells)
                best_area = area
            elif len(cells) == best_size:
                rows = [x[0] for x in cells]
                cols = [x[1] for x in cells]
                area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
                if area > best_area:
                    best_cells = cells
                    best_area = area

    if not best_cells:
        return grid.copy()

    # Extract the template.
    template_mask = np.zeros((h, w), dtype=bool)
    for r, c in best_cells:
        template_mask[r, c] = True

    template_color = int(grid[best_cells[0][0], best_cells[0][1]])

    rows = [r for r, _ in best_cells]
    cols = [c for _, c in best_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    template_h = max_r - min_r + 1
    template_w = max_c - min_c + 1

    template_pat = template_mask[min_r:max_r + 1, min_c:max_c + 1]
    template_cells = [(int(r), int(c)) for r, c in np.argwhere(template_pat)]

    # Pattern dots are all non-background pixels outside the template.
    pattern_cells = []
    for r in range(h):
        cols_here = np.where(nonbg[r] & ~template_mask[r])[0]
        for c in cols_here:
            pattern_cells.append((r, int(c)))

    if not pattern_cells:
        return grid.copy()

    # Pattern color = most frequent non-template color among pattern dots.
    pattern_colors = [int(grid[r, c]) for r, c in pattern_cells]
    non_template_pattern_colors = [x for x in pattern_colors if x != template_color]

    if non_template_pattern_colors:
        uniq, cnt = np.unique(non_template_pattern_colors, return_counts=True)
        pattern_color = int(uniq[np.argmax(cnt)])
    else:
        pattern_color = template_color

    # Choose the center pattern dot.
    pts = np.array(pattern_cells, dtype=float)
    centroid = pts.mean(axis=0)

    def center_key(pt):
        r, c = pt
        dist = (r - centroid[0]) ** 2 + (c - centroid[1]) ** 2
        prefer_template_color = 0 if grid[r, c] == template_color else 1
        return (prefer_template_color, dist, r, c)

    center_r, center_c = min(pattern_cells, key=center_key)

    # Snap coordinates to an even 2-spaced lattice.
    def snap_to_even(x: int) -> int:
        return ((x + 1) // 2) * 2

    center_r_snap = snap_to_even(center_r)
    center_c_snap = snap_to_even(center_c)

    # Build placements: (dr, dc) -> color.
    placements = {}
    for r, c in pattern_cells:
        dr = (snap_to_even(r) - center_r_snap) // 2
        dc = (snap_to_even(c) - center_c_snap) // 2

        key = (dr, dc)
        color = pattern_color if (r, c) == (center_r, center_c) else template_color

        # If two dots snap to the same lattice cell, let the center/pattern color win.
        if key in placements:
            if color == pattern_color and placements[key] != pattern_color:
                placements[key] = pattern_color
        else:
            placements[key] = color

    # Stamp copies into a blank background grid.
    out = np.full((h, w), bg, dtype=grid.dtype)

    # Stamp non-center copies first, center copy last if possible.
    items = list(placements.items())
    items.sort(key=lambda item: (item[1] == pattern_color, item[0][0], item[0][1]))

    for (dr, dc), color in items:
        top_r = min_r + dr * template_h
        top_c = min_c + dc * template_w

        for tr, tc in template_cells:
            r = top_r + tr
            c = top_c + tc
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = color

    return out