import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Sweep a 4-colored line marker perpendicular to itself.

    - Background is the most common border color.
    - The marker is the largest connected component of 4s.
    - If the marker is horizontal, sweep vertically; if vertical, sweep horizontally.
    - Sweep away from the nearest perpendicular border (both directions if centered).
    - At odd distances from the marker, erase the marker cells to background.
    - At even distances, copy the marker:
        * color 4 for pattern positions 0, 1, 3, 4, ...
        * color 3 for pattern positions 2, 5, ...
      On color-3 copy rows/columns, recolor all non-background, non-4 pixels
      in that whole row/column to 3.
    """
    grid = np.array(grid, dtype=int)
    if grid.size == 0:
        return grid

    h, w = grid.shape
    out = grid.copy()

    if not np.any(grid == 4):
        return out

    # Detect background from the border.
    border = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    vals, counts = np.unique(border, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Fallback if the border somehow is mostly 4.
    if bg == 4:
        all_vals, all_counts = np.unique(grid, return_counts=True)
        non4 = all_vals != 4
        if not np.any(non4):
            return out
        bg = int(all_vals[non4][np.argmax(all_counts[non4])])

    # Find the marker: largest 4-component, preferring border-touching on ties.
    visited = np.zeros((h, w), dtype=bool)
    best_cells = []
    best_score = (-1, -1)

    for i in range(h):
        for j in range(w):
            if grid[i, j] == 4 and not visited[i, j]:
                stack = [(i, j)]
                visited[i, j] = True
                cells = []

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and grid[nr, nc] == 4
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                touches_border = any(
                    r == 0 or r == h - 1 or c == 0 or c == w - 1
                    for r, c in cells
                )
                score = (len(cells), int(touches_border))
                if score > best_score:
                    best_score = score
                    best_cells = cells

    if not best_cells:
        return out

    cells = np.array(best_cells, dtype=int)
    rs = cells[:, 0]
    cs = cells[:, 1]

    # Keep the original marker as 4.
    out[rs, cs] = 4

    # Determine marker orientation from its bounding box.
    r_min, r_max = int(rs.min()), int(rs.max())
    c_min, c_max = int(cs.min()), int(cs.max())
    height = r_max - r_min + 1
    width = c_max - c_min + 1
    horizontal = height <= width

    # Sweep away from the nearest perpendicular border.
    if horizontal:
        center = (r_min + r_max) // 2
        dist_up = center
        dist_down = h - 1 - center
        if dist_up < dist_down:
            signs = [1]
        elif dist_down < dist_up:
            signs = [-1]
        else:
            signs = [1, -1]
    else:
        center = (c_min + c_max) // 2
        dist_left = center
        dist_right = w - 1 - center
        if dist_left < dist_right:
            signs = [1]
        elif dist_right < dist_left:
            signs = [-1]
        else:
            signs = [1, -1]

    # Foreground pixels that can be turned into 3 on scan lines.
    fg_mask = (grid != bg) & (grid != 4)

    # Apply the sweep.
    for sign in signs:
        d = 1
        while True:
            if horizontal:
                sr = rs + sign * d
                sc = cs
                valid = (sr >= 0) & (sr < h)
            else:
                sr = rs
                sc = cs + sign * d
                valid = (sc >= 0) & (sc < w)

            if not np.any(valid):
                break

            if d % 2 == 1:
                # Odd distance: erase the marker pattern here.
                out[sr[valid], sc[valid]] = bg
            else:
                # Even distance: stamp the marker pattern here.
                step = d // 2
                if step % 3 == 2:
                    # Color-3 stamp: paint the marker cells and scan the line.
                    out[sr[valid], sc[valid]] = 3

                    if horizontal:
                        for rr in np.unique(sr[valid]):
                            rr = int(rr)
                            out[rr, fg_mask[rr, :]] = 3
                    else:
                        for cc in np.unique(sc[valid]):
                            cc = int(cc)
                            out[fg_mask[:, cc], cc] = 3
                else:
                    # Color-4 stamp: only paint the marker cells.
                    out[sr[valid], sc[valid]] = 4

            d += 1

    return out