import numpy as np
from collections import deque, Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transformation rule
    -------------------
    1. Find the background color (the most common color on the border,
       falling back to the most common color in the whole grid).
    2. Extract all 8-connected non-background objects.
    3. If an object's pixel count is shared by at least one other object,
       move it to the top edge. Otherwise move it to the bottom edge.
    4. Objects keep their shape and horizontal order.  If several objects
       are moved to the same edge and their original horizontal intervals
       overlap, they are packed left-to-right without overlap while
       preserving their original left-to-right order as much as possible.
    """
    if grid is None or grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # Background color: prefer the dominant border color, otherwise global mode.
    border = np.concatenate([
        grid[0, :],
        grid[-1, :],
        grid[:, 0],
        grid[:, -1]
    ])
    bg = int(np.bincount(border.astype(int, copy=False).ravel()).argmax())

    # If the border mode is not actually present in the grid (defensive),
    # use the global mode.
    if bg not in np.unique(grid):
        bg = int(np.bincount(grid.astype(int, copy=False).ravel()).argmax())

    non_bg = grid != bg
    if not np.any(non_bg):
        return grid.copy()

    # ---------------------------------------------------------------
    # 8-connected component extraction
    # ---------------------------------------------------------------
    visited = np.zeros((H, W), dtype=bool)
    components = []

    for r in range(H):
        for c in range(W):
            if visited[r, c] or grid[r, c] == bg:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True

            cells = []
            min_r, max_r = r, r
            min_c, max_c = c, c

            while q:
                y, x = q.popleft()
                cells.append((y, x))

                min_r = min(min_r, y)
                max_r = max(max_r, y)
                min_c = min(min_c, x)
                max_c = max(max_c, x)

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if not visited[ny, nx] and grid[ny, nx] == color:
                                visited[ny, nx] = True
                                q.append((ny, nx))

            rel_cells = [(y - min_r, x - min_c) for (y, x) in cells]

            components.append({
                "color": color,
                "size": len(cells),
                "height": max_r - min_r + 1,
                "width": max_c - min_c + 1,
                "min_r": min_r,
                "max_r": max_r,
                "min_c": min_c,
                "max_c": max_c,
                "rel_cells": rel_cells,
            })

    if not components:
        return grid.copy()

    # ---------------------------------------------------------------
    # Decide top/bottom: non-unique sizes go top, unique sizes go bottom
    # ---------------------------------------------------------------
    size_counts = Counter(comp["size"] for comp in components)
    for comp in components:
        comp["top"] = size_counts[comp["size"]] >= 2

    # ---------------------------------------------------------------
    # Place components
    # ---------------------------------------------------------------
    out = np.full_like(grid, bg, dtype=grid.dtype)

    def place_band(is_top: bool):
        band = [comp for comp in components if comp["top"] == is_top]
        # Preserve original horizontal order.
        band.sort(key=lambda z: (z["min_c"], z["max_c"], z["min_r"], z["size"], z["color"]))

        prev_end_col = -1

        for comp in band:
            width = comp["width"]
            height = comp["height"]

            x = comp["min_c"]

            # Avoid horizontal overlap inside the same band.
            if prev_end_col >= 0:
                x = max(x, prev_end_col + 1)

            if x < 0:
                x = 0
            if x > W - width:
                x = W - width

            prev_end_col = max(prev_end_col, x + width - 1)

            if is_top:
                y0 = 0
            else:
                y0 = H - height

            for dy, dx in comp["rel_cells"]:
                ny, nx = y0 + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    out[ny, nx] = comp["color"]

    place_band(True)
    place_band(False)

    return out