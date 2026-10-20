import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a panelized ARC grid:
      - Separator lines of 1s split the grid into 4 panels.
      - One panel contains the shape to repeat (usually color 4).
      - One panel contains marker objects (usually color 3); their count is the repeat count.
      - The two solid panels provide foreground and background colors.
    The cropped shape mask is repeated with a one-cell background gap,
    horizontally for vertical separators and vertically for horizontal separators.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    H_grid, W_grid = grid.shape

    # Find full separator lines of color 1.
    full_rows = np.where(np.all(grid == 1, axis=1))[0].tolist()
    full_cols = np.where(np.all(grid == 1, axis=0))[0].tolist()

    if not full_rows and not full_cols:
        return np.zeros_like(grid, dtype=int)

    # Choose orientation based on which full lines exist.
    if len(full_cols) >= len(full_rows):
        axis = "cols"
        seps = full_cols
    else:
        axis = "rows"
        seps = full_rows

    # Split into panels between separator lines.
    panels = []
    if axis == "cols":
        start = 0
        for s in seps:
            if s > start:
                panels.append(grid[:, start:s].copy())
            start = s + 1
        if start < W_grid:
            panels.append(grid[:, start:W_grid].copy())
    else:
        start = 0
        for s in seps:
            if s > start:
                panels.append(grid[start:s, :].copy())
            start = s + 1
        if start < H_grid:
            panels.append(grid[start:H_grid, :].copy())

    panels = [p for p in panels if p.size > 0]
    if len(panels) < 4:
        return np.zeros_like(grid, dtype=int)

    def unique_values(panel):
        return np.unique(panel)

    # Identify solid panels: uniform nonzero non-separator panels.
    solid_indices = []
    for i, panel in enumerate(panels):
        vals = unique_values(panel)
        if vals.size == 1 and vals[0] not in (0, 1):
            solid_indices.append(i)

    if len(solid_indices) < 2:
        if len(panels) >= 2:
            solid_indices = [len(panels) - 2, len(panels) - 1]
        else:
            return np.zeros_like(grid, dtype=int)

    # The two solid panels closest to the end are foreground and background.
    fg_color = int(panels[solid_indices[-2]].flat[0])
    bg_color = int(panels[solid_indices[-1]].flat[0])

    # Identify marker panel (color 3 with zeros) and shape panel (color 4 with zeros).
    pattern_idx = None
    template_idx = None

    for i, panel in enumerate(panels):
        vals = set(unique_values(panel).tolist())
        if 3 in vals and 0 in vals and len(vals) > 1:
            pattern_idx = i
            break

    for i, panel in enumerate(panels):
        vals = set(unique_values(panel).tolist())
        if 4 in vals and 0 in vals and len(vals) > 1:
            template_idx = i
            break

    # Fallbacks for unusual cases.
    if template_idx is None or pattern_idx is None:
        non_solid = [i for i in range(len(panels)) if i not in set(solid_indices)]
        if len(non_solid) >= 2:
            if template_idx is None:
                template_idx = non_solid[0]
            if pattern_idx is None:
                for idx in non_solid:
                    if idx != template_idx:
                        pattern_idx = idx
                        break

    if template_idx is None:
        template_idx = 0
    if pattern_idx is None:
        pattern_idx = 1 if len(panels) > 1 else 0

    template_panel = panels[template_idx]
    pattern_panel = panels[pattern_idx]

    # Extract shape mask and crop to its bounding box.
    if np.any(template_panel == 4):
        mask = template_panel == 4
    else:
        mask = template_panel != 0

    if not np.any(mask):
        mask = template_panel != 0

    row_any = np.any(mask, axis=1)
    col_any = np.any(mask, axis=0)

    if not np.any(row_any) or not np.any(col_any):
        return np.zeros_like(grid, dtype=int)

    r0 = int(np.argmax(row_any))
    r1 = int(len(row_any) - 1 - np.argmax(row_any[::-1]))
    c0 = int(np.argmax(col_any))
    c1 = int(len(col_any) - 1 - np.argmax(col_any[::-1]))

    mask = mask[r0:r1 + 1, c0:c1 + 1]

    # Extract marker mask.
    if np.any(pattern_panel == 3):
        marker_mask = pattern_panel == 3
    else:
        marker_mask = pattern_panel != 0

    def count_connected_components(m):
        """Count 4-connected components in a boolean mask."""
        h, w = m.shape
        visited = np.zeros((h, w), dtype=bool)
        count = 0

        for i in range(h):
            for j in range(w):
                if m[i, j] and not visited[i, j]:
                    count += 1
                    visited[i, j] = True
                    stack = [(i, j)]

                    while stack:
                        y, x = stack.pop()

                        if y > 0 and m[y - 1, x] and not visited[y - 1, x]:
                            visited[y - 1, x] = True
                            stack.append((y - 1, x))

                        if y + 1 < h and m[y + 1, x] and not visited[y + 1, x]:
                            visited[y + 1, x] = True
                            stack.append((y + 1, x))

                        if x > 0 and m[y, x - 1] and not visited[y, x - 1]:
                            visited[y, x - 1] = True
                            stack.append((y, x - 1))

                        if x + 1 < w and m[y, x + 1] and not visited[y, x + 1]:
                            visited[y, x + 1] = True
                            stack.append((y, x + 1))

        return count

    repeats = count_connected_components(marker_mask)

    # Graceful fallback if marker counting fails for some reason.
    if repeats <= 0:
        total_markers = int(np.count_nonzero(marker_mask))
        repeats = total_markers // 2
    if repeats <= 0:
        repeats = 1

    sh, sw = mask.shape

    # Build one block: foreground where mask is true, background elsewhere.
    block = np.full((sh, sw), bg_color, dtype=int)
    block[mask] = fg_color

    if axis == "cols":
        # Vertical separators -> repeat horizontally.
        out_h = sh
        out_w = repeats * sw + (repeats - 1)
        out = np.full((out_h, out_w), bg_color, dtype=int)

        for k in range(repeats):
            start_col = k * (sw + 1)
            out[:, start_col:start_col + sw] = block

    else:
        # Horizontal separators -> repeat vertically.
        out_w = sw
        out_h = repeats * sh + (repeats - 1)
        out = np.full((out_h, out_w), bg_color, dtype=int)

        for k in range(repeats):
            start_row = k * (sh + 1)
            out[start_row:start_row + sh, :] = block

    return out