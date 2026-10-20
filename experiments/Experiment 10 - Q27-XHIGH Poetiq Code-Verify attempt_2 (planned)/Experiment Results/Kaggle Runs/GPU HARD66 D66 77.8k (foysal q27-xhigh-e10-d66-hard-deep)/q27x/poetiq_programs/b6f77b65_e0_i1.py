import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:
    1. The pixel at (0, 0) is a "key" color.
    2. Remove every other cell in the grid that has this key color.
    3. Each remaining 4-connected same-color component falls downward as far
       as possible without leaving the grid and without overlapping another
       already-settled component.
    4. Components are settled from bottom to top.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    key = grid[0, 0]

    # Work on a copy and remove the key color except the key pixel itself.
    out = grid.copy()
    mask = (out == key)
    mask[0, 0] = False
    out[mask] = 0

    # Nonzero cells are the remaining cells.
    occupied = out != 0

    # Label 4-connected components of same color.
    labeled, n = ndimage.label(occupied)

    if n == 0:
        # Keep only the key pixel.
        ans = np.zeros_like(out)
        ans[0, 0] = key
        return ans

    # Build component records.
    comps = []
    for cid in range(1, n + 1):
        ys, xs = np.where(labeled == cid)
        color = out[ys[0], xs[0]]
        cells = set(zip(ys.tolist(), xs.tolist()))
        min_y, max_y = ys.min(), ys.max()
        comps.append({
            "cells": cells,
            "color": color,
            "min_y": min_y,
            "max_y": max_y,
            "settled": None
        })

    # Settle bottom-most components first.
    order = sorted(range(n), key=lambda i: comps[i]["max_y"], reverse=True)

    settled_grid = np.zeros((H, W), dtype=bool)

    def can_place(cells, dy):
        for (y, x) in cells:
            ny = y + dy
            if ny < 0 or ny >= H:
                return False
            if settled_grid[ny, x]:
                return False
        return True

    for idx in order:
        comp = comps[idx]
        # Find max downward shift.
        best = 0
        for dy in range(H - comp["min_y"] + 1):
            if can_place(comp["cells"], dy):
                best = dy
            else:
                break

        # Place component.
        for (y, x) in comp["cells"]:
            ny = y + best
            settled_grid[ny, x] = True
            out[ny, x] = comp["color"]
            out[y, x] = 0

    # Restore the key pixel.
    out[0, 0] = key
    return out