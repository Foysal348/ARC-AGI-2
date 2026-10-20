import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid according to the observed ARC rule.

    Rule:
      1. The top-left cell contains a “key” color.
      2. All other cells having that key color are removed.
      3. The remaining non-zero cells are split into 4-connected components.
      4. Components are settled from bottom to top.
      5. Each component falls straight downward as far as possible without
         leaving the grid or colliding with already settled components.
      6. Small 4-cell L-corner pieces that are directly side-adjacent to a
         removed key cell are first flipped vertically before settling.

    This reproduces the supplied examples: the key-colored objects disappear,
    nearby L-corners may flip, and the remaining shapes drop into the space
    below them, preserving left-right positions.
    """
    if grid.size == 0:
        return grid.copy()

    g = np.array(grid, copy=True)
    h, w = g.shape

    key = int(g[0, 0])

    # ------------------------------------------------------------------
    # 1) Remove all key-colored cells except the fixed top-left marker.
    # ------------------------------------------------------------------
    out = np.zeros((h, w), dtype=g.dtype)
    out[0, 0] = key

    removed = np.zeros((h, w), dtype=bool)
    removed[g == key] = True
    removed[0, 0] = False

    # Working occupancy for the final answer.
    occupied = np.zeros((h, w), dtype=bool)
    occupied[0, 0] = True

    # ------------------------------------------------------------------
    # 2) Connected components of the remaining non-zero cells.
    #    Use 4-connectivity.
    # ------------------------------------------------------------------
    remaining = (g != 0) & (~removed)

    comp_id = np.zeros((h, w), dtype=int)
    components = []

    for r in range(h):
        for c in range(w):
            if remaining[r, c] and comp_id[r, c] == 0:
                comp_id[r, c] = 1
                cells = [(r, c)]
                stack = [(r, c)]

                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if remaining[ny, nx] and comp_id[ny, nx] == 0:
                                comp_id[ny, nx] = 1
                                stack.append((ny, nx))
                                cells.append((ny, nx))

                components.append({
                    "cells": cells,
                    "color": int(g[r, c])
                })

    # ------------------------------------------------------------------
    # Helpers for component shape / placement.
    # ------------------------------------------------------------------
    def side_adjacent_to_removed(cells):
        """
        True if any cell of the component is horizontally adjacent
        to a removed key cell.
        """
        for r, c in cells:
            if c > 0 and removed[r, c - 1]:
                return True
            if c + 1 < w and removed[r, c + 1]:
                return True
        return False

    def is_small_l_corner(cells):
        """
        Recognize the 4-cell L-corner shapes used in the examples:
        bounding box 2x3 or 3x2 with exactly 4 cells.
        """
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        hh = max(rows) - min(rows) + 1
        ww = max(cols) - min(cols) + 1
        return len(cells) == 4 and ((hh == 2 and ww == 3) or (hh == 3 and ww == 2))

    def normalize(cells):
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        return sorted((r - min_r, c - min_c) for r, c in cells)

    def flip_vertical(cells):
        """
        Flip a shape vertically inside its bounding box.
        This matches the observed behavior of the small L-corners.
        """
        norm = normalize(cells)
        hh = max(r for r, _ in norm) + 1
        ww = max(c for _, c in norm) + 1
        return sorted((hh - 1 - r, c) for r, c in norm)

    def can_place(norm_cells, top_r, left_c):
        for dr, dc in norm_cells:
            r, c = top_r + dr, left_c + dc
            if r < 0 or r >= h or c < 0 or c >= w:
                return False
            if occupied[r, c]:
                return False
        return True

    def place(norm_cells, top_r, left_c, color):
        for dr, dc in norm_cells:
            r, c = top_r + dr, left_c + dc
            occupied[r, c] = True
            out[r, c] = color

    # ------------------------------------------------------------------
    # 3) Settle components from bottom to top.
    # ------------------------------------------------------------------
    components.sort(key=lambda comp: -max(r for r, _ in comp["cells"]))

    for comp in components:
        cells = comp["cells"]
        color = comp["color"]

        # If a small L-corner is side-adjacent to a removed key cell,
        # flip it vertically before falling.
        if is_small_l_corner(cells) and side_adjacent_to_removed(cells):
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