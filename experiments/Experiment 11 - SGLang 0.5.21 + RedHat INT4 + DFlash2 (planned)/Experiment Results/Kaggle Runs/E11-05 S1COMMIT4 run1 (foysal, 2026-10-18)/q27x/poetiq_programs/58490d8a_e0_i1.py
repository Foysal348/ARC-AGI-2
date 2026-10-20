import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:
    1. The input contains a rectangular "key" area whose background color is 0.
       In all examples, all 0-cells belong to this key rectangle, so its
       bounding box can be found from all zero cells.
    2. The non-zero cells inside that key rectangle are a vertical list of colors
       to report, ordered top-to-bottom.
    3. For each listed color, count the number of 8-connected components of that
       color in the rest of the grid, excluding the key rectangle.
    4. The output is the key rectangle filled with 0s. In the row corresponding
       to each listed color, place that many pixels of that color, starting at the
       original legend column and spaced every other column.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Find the bounding box of the zero-background key region.
    zero_mask = (grid == 0)
    if not zero_mask.any():
        return np.zeros_like(grid)

    row_idxs = np.where(zero_mask.any(axis=1))[0]
    col_idxs = np.where(zero_mask.any(axis=0))[0]

    r0, r1 = int(row_idxs[0]), int(row_idxs[-1])
    c0, c1 = int(col_idxs[0]), int(col_idxs[-1])

    key_h = r1 - r0 + 1
    key_w = c1 - c0 + 1

    # Mark cells outside the key region as valid for object counting.
    valid = np.ones((H, W), dtype=bool)
    valid[r0:r1 + 1, c0:c1 + 1] = False

    # Count 8-connected components for every non-zero color outside the key.
    counts = {}
    visited = np.zeros((H, W), dtype=bool)
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for r in range(H):
        for c in range(W):
            if not valid[r, c]:
                continue

            color = grid[r, c]
            if color == 0 or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            component_count = 0

            while stack:
                rr, cc = stack.pop()
                component_count += 1

                for dr, dc in offsets:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < H and 0 <= nc < W and
                        valid[nr, nc] and not visited[nr, nc] and
                        grid[nr, nc] == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            color = int(color)
            counts[color] = counts.get(color, 0) + 1

    # Extract the legend colors inside the key region, preserving vertical order.
    legend = []
    seen_colors = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            color = int(grid[r, c])
            if color != 0 and color not in seen_colors:
                legend.append((color, r - r0, c - c0))
                seen_colors.add(color)

    # If the key rectangle is unexpectedly narrow for the required marks,
    # widen the output just enough to display all counted marks.
    max_col_needed = 0
    for color, lr, lc in legend:
        n = counts.get(color, 0)
        if n > 0:
            max_col_needed = max(max_col_needed, lc + 2 * (n - 1))

    out_w = max(key_w, max_col_needed + 1)
    out = np.zeros((key_h, out_w), dtype=grid.dtype)

    # Fill output rows with the counted number of legend-color pixels.
    for color, lr, lc in legend:
        n = counts.get(color, 0)
        for i in range(n):
            col = lc + 2 * i
            out[lr, col] = color

    return out