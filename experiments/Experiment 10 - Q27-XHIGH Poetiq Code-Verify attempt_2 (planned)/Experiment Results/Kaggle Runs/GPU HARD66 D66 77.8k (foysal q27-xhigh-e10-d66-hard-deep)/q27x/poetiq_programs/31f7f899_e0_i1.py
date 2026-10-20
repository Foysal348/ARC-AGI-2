import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Sort and re-center vertical bars around a horizontal baseline.

    The input contains a horizontal line (color 6 in the given examples).
    Each other connected non-background component that crosses this line is
    treated as a vertical bar. For every bar, we measure its reach from the
    baseline row as the maximum vertical distance to any of its cells.
    These reaches are sorted ascending and assigned to bars from left to right.
    Finally, each bar is redrawn centered on the baseline row with its new reach.
    """
    grid = np.array(grid, dtype=int, copy=True)

    if grid.ndim != 2 or grid.size == 0:
        return grid

    h, w = grid.shape

    # Background color: the most frequent color.
    flat = grid.ravel()
    if flat.min() >= 0:
        bg = int(np.bincount(flat).argmax())
    else:
        vals, cnts = np.unique(flat, return_counts=True)
        bg = int(vals[np.argmax(cnts)])

    # Locate the horizontal baseline.
    # The examples use color 6, so prefer that if present.
    baseline_row = None
    baseline_color = None

    if np.any(grid == 6):
        counts6 = np.count_nonzero(grid == 6, axis=1)
        if counts6.max() > 0:
            baseline_row = int(np.argmax(counts6))
            baseline_color = 6

    # Fallback for variation: use the row with the most non-background cells.
    if baseline_row is None:
        nonbg_counts = np.count_nonzero(grid != bg, axis=1)
        if nonbg_counts.max() == 0:
            return grid

        baseline_row = int(np.argmax(nonbg_counts))
        row_vals, row_counts = np.unique(grid[baseline_row], return_counts=True)
        nonbg_mask = row_vals != bg

        if not np.any(nonbg_mask):
            return grid

        baseline_color = int(row_vals[nonbg_mask][np.argmax(row_counts[nonbg_mask])])

    if baseline_color == bg:
        return grid

    # Find monochrome connected components of non-background, non-baseline colors.
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue

            color = int(grid[r, c])

            if color == bg or color == baseline_color:
                visited[r, c] = True
                continue

            # BFS for this component.
            stack = [(r, c)]
            visited[r, c] = True

            cells = []
            min_col = c
            max_dist = 0
            baseline_cols = set()

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))

                if cc < min_col:
                    min_col = cc

                dist = abs(rr - baseline_row)
                if dist > max_dist:
                    max_dist = dist

                if rr == baseline_row:
                    baseline_cols.add(cc)

                # 4-connected neighbors.
                if rr > 0 and not visited[rr - 1, cc] and grid[rr - 1, cc] == color:
                    visited[rr - 1, cc] = True
                    stack.append((rr - 1, cc))

                if rr + 1 < h and not visited[rr + 1, cc] and grid[rr + 1, cc] == color:
                    visited[rr + 1, cc] = True
                    stack.append((rr + 1, cc))

                if cc > 0 and not visited[rr, cc - 1] and grid[rr, cc - 1] == color:
                    visited[rr, cc - 1] = True
                    stack.append((rr, cc - 1))

                if cc + 1 < w and not visited[rr, cc + 1] and grid[rr, cc + 1] == color:
                    visited[rr, cc + 1] = True
                    stack.append((rr, cc + 1))

            # Only components crossing the baseline are vertical bars.
            if baseline_cols:
                components.append((min_col, color, max_dist, cells, baseline_cols))

    if not components:
        return grid

    # Sort bars left-to-right, and sort their reaches independently.
    components.sort(key=lambda x: x[0])
    new_distances = sorted(comp[2] for comp in components)

    out = grid.copy()

    # Clear old vertical bars.
    for _, _, _, cells, _ in components:
        for rr, cc in cells:
            out[rr, cc] = bg

    # Redraw each bar centered on the baseline row with its assigned distance.
    for (_, color, _, _, baseline_cols), new_d in zip(components, new_distances):
        top = max(0, baseline_row - new_d)
        bottom = min(h - 1, baseline_row + new_d)

        for cc in sorted(baseline_cols):
            out[top:bottom + 1, cc] = color

    return out