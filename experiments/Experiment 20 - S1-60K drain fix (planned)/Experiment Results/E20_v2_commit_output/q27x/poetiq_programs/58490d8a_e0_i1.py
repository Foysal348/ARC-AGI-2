import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC grid by summarizing colored objects using a zero rectangle.

    Rule:
    1. The zero cells form a rectangular canvas. Its bounding box is the output shape.
    2. Every non-zero cell inside that canvas is a key:
         - its relative row is the output row,
         - its color is the output color.
    3. For each key color, count the 8-connected components of that color
       outside the zero canvas.
    4. In the output canvas, place that many pixels of the key color in the
       key's row, starting at the key's column and moving right by 2 columns.
    """
    grid = np.asarray(grid)

    # Gracefully handle invalid / empty inputs.
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    zero_mask = (grid == 0)
    if not zero_mask.any():
        return grid.copy()

    # Bounding box of all zero pixels: this is the zero canvas / output size.
    zero_rows = np.where(zero_mask.any(axis=1))[0]
    zero_cols = np.where(zero_mask.any(axis=0))[0]

    r0, r1 = int(zero_rows[0]), int(zero_rows[-1])
    c0, c1 = int(zero_cols[0]), int(zero_cols[-1])

    # In the normal case, all zero pixels belong to the canvas.
    # If there are stray isolated zero pixels, fall back to the largest
    # 4-connected zero component to avoid expanding the canvas incorrectly.
    all_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    zero_count = int(zero_mask.sum())

    if all_area > 0 and (zero_count / all_area) < 0.5:
        visited_zero = np.zeros((H, W), dtype=bool)
        best_bbox = None
        best_size = -1

        for i in range(H):
            for j in range(W):
                if zero_mask[i, j] and not visited_zero[i, j]:
                    stack = [(i, j)]
                    visited_zero[i, j] = True

                    area = 0
                    min_r = max_r = i
                    min_c = max_c = j

                    while stack:
                        r, c = stack.pop()
                        area += 1

                        min_r = min(min_r, r)
                        max_r = max(max_r, r)
                        min_c = min(min_c, c)
                        max_c = max(max_c, c)

                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < H
                                and 0 <= nc < W
                                and zero_mask[nr, nc]
                                and not visited_zero[nr, nc]
                            ):
                                visited_zero[nr, nc] = True
                                stack.append((nr, nc))

                    bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
                    if bbox_area > best_size:
                        best_size = bbox_area
                        best_bbox = (min_r, max_r, min_c, max_c)

        if best_bbox is not None:
            r0, r1, c0, c1 = best_bbox

    out_h = r1 - r0 + 1
    out_w = c1 - c0 + 1
    out = np.zeros((out_h, out_w), dtype=grid.dtype)

    # Mark the canvas region.
    canvas = np.zeros((H, W), dtype=bool)
    canvas[r0:r1 + 1, c0:c1 + 1] = True
    outside = ~canvas

    # Collect all keys inside the canvas.
    legend = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            val = grid[r, c]
            if val != 0:
                legend.append((r - r0, c - c0, int(val)))

    if not legend:
        return out

    count_cache = {}

    def count_8_components(color: int) -> int:
        """Count 8-connected components of `color` outside the canvas."""
        if color in count_cache:
            return count_cache[color]

        mask = (grid == color) & outside
        if not mask.any():
            count_cache[color] = 0
            return 0

        visited = np.zeros((H, W), dtype=bool)
        count = 0

        for y, x in zip(*np.where(mask)):
            if visited[y, x]:
                continue

            count += 1
            stack = [(y, x)]
            visited[y, x] = True

            while stack:
                cy, cx = stack.pop()

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue

                        ny, nx = cy + dy, cx + dx

                        if (
                            0 <= ny < H
                            and 0 <= nx < W
                            and mask[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))

        count_cache[color] = count
        return count

    # Place summary dots for each legend key.
    for rel_r, rel_c, color in legend:
        n_components = count_8_components(color)
        col = rel_c

        while n_components > 0 and 0 <= rel_r < out_h and 0 <= col < out_w:
            out[rel_r, col] = color
            n_components -= 1
            col += 2

    return out