import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the ARC grid by:
      1. Keeping the top-left cell fixed.
      2. Removing all other cells whose color equals the top-left marker color.
      3. Splitting the remaining nonzero cells into 4-connected components.
      4. Vertically flipping small L-corner components that are horizontally
         adjacent to a removed cell.
      5. Settling components from bottom to top, moving each rigidly downward
         as far as possible without leaving the grid or colliding with already
         settled components.
    """
    if grid.size == 0:
        return grid.copy()

    grid = np.array(grid, copy=True)
    h, w = grid.shape
    key = int(grid[0, 0])

    out = np.zeros((h, w), dtype=grid.dtype)
    out[0, 0] = key

    # Mark removed key-colored cells, except the fixed top-left marker.
    removed = np.zeros((h, w), dtype=bool)
    removed[grid == key] = True
    removed[0, 0] = False

    remaining = (grid != 0) & (~removed)

    # 4-connected component extraction.
    comp_id = np.zeros((h, w), dtype=int)
    components = []

    for r in range(h):
        for c in range(w):
            if remaining[r, c] and comp_id[r, c] == 0:
                cells = []
                stack = [(r, c)]
                comp_id[r, c] = 1
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if remaining[ny, nx] and comp_id[ny, nx] == 0:
                                comp_id[ny, nx] = 1
                                stack.append((ny, nx))

                components.append({
                    "cells": cells,
                    "color": int(grid[r, c]),
                })

    def normalize(cells):
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        return sorted((r - min_r, c - min_c) for r, c in cells)

    def flip_vertical(cells):
        norm = normalize(cells)
        hh = max(r for r, _ in norm) + 1
        return sorted((hh - 1 - r, c) for r, c in norm)

    def is_small_l_corner(cells):
        hh = max(r for r, _ in cells) - min(r for r, _ in cells) + 1
        ww = max(c for _, c in cells) - min(c for _, c in cells) + 1
        return len(cells) == 4 and ((hh == 2 and ww == 3) or (hh == 3 and ww == 2))

    def horizontally_adjacent_to_removed(cells):
        cell_set = set(cells)
        for r, c in cells:
            for nr, nc in ((r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and removed[nr, nc]:
                    return True
        return False

    occupied = np.zeros((h, w), dtype=bool)
    occupied[0, 0] = True

    def can_place(shape, top_r, left_c):
        for dr, dc in shape:
            r, c = top_r + dr, left_c + dc
            if r < 0 or r >= h or c < 0 or c >= w:
                return False
            if occupied[r, c]:
                return False
        return True

    def place(shape, top_r, left_c, color):
        for dr, dc in shape:
            r, c = top_r + dr, left_c + dc
            occupied[r, c] = True
            out[r, c] = color

    # Settle from bottom to top; ties by leftmost column for determinism.
    components.sort(key=lambda comp: (-max(r for r, _ in comp["cells"]),
                                      min(c for _, c in comp["cells"])))

    for comp in components:
        cells = comp["cells"]
        color = comp["color"]

        if is_small_l_corner(cells) and horizontally_adjacent_to_removed(cells):
            shape = flip_vertical(cells)
        else:
            shape = normalize(cells)

        base_r = min(r for r, _ in cells)
        base_c = min(c for _, c in cells)
        max_dr = max(dr for dr, _ in shape)
        max_off = h - 1 - (base_r + max_dr)

        final_off = 0
        for off in range(max_off, -1, -1):
            if can_place(shape, base_r + off, base_c):
                final_off = off
                break

        place(shape, base_r + final_off, base_c, color)

    return out