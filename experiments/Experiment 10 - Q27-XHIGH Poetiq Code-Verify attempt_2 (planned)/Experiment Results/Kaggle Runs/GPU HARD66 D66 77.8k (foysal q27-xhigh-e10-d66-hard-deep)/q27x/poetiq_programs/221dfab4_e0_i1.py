import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid by copying a 4-colored line marker into the grid.

    Rule:
    - Find the largest connected component of color 4. It is assumed to be a
      horizontal or vertical line segment.
    - Propagate copies of it every 2 cells in the direction away from the
      nearest edge along the axis perpendicular to the marker.
    - Copy colors repeat as 4, 4, 3.
    - On 3-colored copies, additionally recolor all non-background, non-4
      pixels in the affected row or column to 3.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # Detect background as the most common border color.
    border = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    vals, counts = np.unique(border, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Fallback if border is mostly color 4.
    if bg == 4:
        all_vals, all_counts = np.unique(grid, return_counts=True)
        non4 = all_vals != 4
        if not np.any(non4):
            return grid.copy()
        bg = int(all_vals[non4][np.argmax(all_counts[non4])])

    out = grid.copy()

    # If there is no marker, nothing to do.
    if not np.any(grid == 4):
        return out

    marker = grid == 4

    # Find the largest border-touching / largest 4-component.
    visited = np.zeros((h, w), dtype=bool)
    best_cells = []
    best_key = (False, 0)  # (touches_border, size)

    for i in range(h):
        for j in range(w):
            if marker[i, j] and not visited[i, j]:
                stack = [(i, j)]
                visited[i, j] = True
                cells = []

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < h and 0 <= nc < w
                            and marker[nr, nc]
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                if not cells:
                    continue

                touches_border = any(
                    r == 0 or r == h - 1 or c == 0 or c == w - 1
                    for r, c in cells
                )
                key = (touches_border, len(cells))
                if key > best_key:
                    best_key = key
                    best_cells = cells

    if not best_cells:
        return out

    cells = np.array(best_cells, dtype=int)
    r_min, r_max = cells[:, 0].min(), cells[:, 0].max()
    c_min, c_max = cells[:, 1].min(), cells[:, 1].max()

    marker_h = r_max - r_min + 1
    marker_w = c_max - c_min + 1

    # Determine marker orientation and propagation direction.
    if marker_h <= marker_w:
        # Horizontal marker: propagate vertically.
        center_row = (r_min + r_max) // 2
        direction = 1 if center_row < h - 1 - center_row else -1
        dr, dc = direction, 0
    else:
        # Vertical marker: propagate horizontally.
        center_col = (c_min + c_max) // 2
        direction = 1 if center_col < w - 1 - center_col else -1
        dr, dc = 0, direction

    # Pixels that can be recolored to 3 on "3" scan lines.
    fg = (grid != bg) & (grid != 4)

    d = 0
    max_d = max(h, w) * 2 + 10

    while d <= max_d:
        sr = cells[:, 0] + dr * d
        sc = cells[:, 1] + dc * d

        valid = (sr >= 0) & (sr < h) & (sc >= 0) & (sc < w)
        if not np.any(valid):
            break

        step_index = d // 2
        color = 3 if step_index % 3 == 2 else 4

        if color == 4:
            out[sr[valid], sc[valid]] = 4
        else:
            out[sr[valid], sc[valid]] = 3

            # On 3-colored copies, recolor all foreground pixels in the
            # affected row(s) or column(s).
            if dr != 0:
                for rr in np.unique(sr[valid]):
                    rr = int(rr)
                    if 0 <= rr < h:
                        cols = np.where(fg[rr, :])[0]
                        if cols.size:
                            out[rr, cols] = 3
            else:
                for cc in np.unique(sc[valid]):
                    cc = int(cc)
                    if 0 <= cc < w:
                        rows = np.where(fg[:, cc])[0]
                        if rows.size:
                            out[rows, cc] = 3

        d += 2

    return out