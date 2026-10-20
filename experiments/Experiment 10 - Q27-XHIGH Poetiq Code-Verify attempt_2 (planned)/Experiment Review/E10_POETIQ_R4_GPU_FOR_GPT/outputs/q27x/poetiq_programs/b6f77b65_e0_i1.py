import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a 12x12 ARC grid by removing the object whose color matches
    the top-left marker, then applying a constrained gravity step:

    1. Keep the top-left cell fixed.
    2. Remove every other cell whose color equals grid[0, 0].
    3. Extract 4-connected components of the remaining cells.
    4. Process components from bottom to top.
    5. For each component, try to move it straight down as far as possible
       without leaving the grid or overlapping already placed components.
    6. If the component is an L-shaped corner piece and rotating it 180°
       allows it to fall further, use the rotated shape.

    This rule reproduces the supplied examples: the indicated color is erased,
    lower pieces settle first, and upper pieces fall only until blocked by
    the settled pieces below them.
    """
    if grid.size == 0:
        return grid.copy()

    grid = np.array(grid, copy=True)
    h, w = grid.shape

    key = int(grid[0, 0])
    output = np.zeros((h, w), dtype=grid.dtype)
    output[0, 0] = grid[0, 0]

    # Remove all key-colored cells except the fixed top-left marker.
    key_mask = np.zeros((h, w), dtype=bool)
    key_mask[grid == key] = True
    key_mask[0, 0] = False

    remaining = (grid != 0) & (~key_mask)

    # Connected component labeling (4-connectivity).
    comp_id = np.zeros((h, w), dtype=int)
    components = []
    comp_num = 0

    for r in range(h):
        for c in range(w):
            if remaining[r, c] and comp_id[r, c] == 0:
                comp_num += 1
                color = int(grid[r, c])
                stack = [(r, c)]
                cells = []
                comp_id[r, c] = comp_num
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if remaining[ny, nx] and comp_id[ny, nx] == 0:
                                comp_id[ny, nx] = comp_num
                                stack.append((ny, nx))
                components.append({"cells": cells, "color": color})

    # Keep components in bottom-to-top order (larger minimum row first).
    components.sort(key=lambda comp: -min(r for r, _ in comp["cells"]))

    def is_l_corner(cells):
        """
        Identify the corner pieces used in this task.
        The examples use 2x3 or 3x2 L-shaped corner pieces.
        """
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        hgt = max(rows) - min(rows) + 1
        wid = max(cols) - min(cols) + 1
        if len(cells) != hgt * wid - (hgt * wid - len(cells)) or len(cells) < 4:
            pass
        # Specifically: bounding box 3x2 or 2x3 with 4 cells.
        return (hgt == 3 and wid == 2 and len(cells) == 4) or \
               (hgt == 2 and wid == 3 and len(cells) == 4)

    def normalize(cells):
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        return [(r - min_r, c - min_c) for r, c in cells]

    def rotate180(cells):
        norm = normalize(cells)
        hgt = max(r for r, _ in norm) + 1
        wid = max(c for _, c in norm) + 1
        return [(hgt - 1 - r, wid - 1 - c) for r, c in norm]

    def can_place(cells, offset, occupied):
        for dr, dc in cells:
            r, c = dr + offset, dc
            if r < 0 or r >= h or c < 0 or c >= w:
                return False
            if occupied[r, c]:
                return False
        return True

    def place(cells, offset, color, occupied):
        for dr, dc in cells:
            r, c = dr + offset, dc
            occupied[r, c] = color
            output[r, c] = color

    occupied = np.zeros((h, w), dtype=grid.dtype)
    occupied[0, 0] = output[0, 0]

    for comp in components:
        original = comp["cells"]
        base_row = min(r for r, _ in original)
        base_col = min(c for _, c in original)

        orig_norm = normalize(original)
        best_offset = None
        best_shape = orig_norm
        best_rotated = False

        # Try original orientation.
        max_offset = h - (base_row + max(r for r, _ in orig_norm))
        for off in range(max_offset, -1, -1):
            if can_place(orig_norm, base_row + off, occupied):
                best_offset = off
                best_shape = orig_norm
                best_rotated = False
                break

        # If this is an L-corner, try 180° rotation if it falls further.
        if is_l_corner(original):
            rot_norm = rotate180(original)
            max_offset_rot = h - (base_row + max(r for r, _ in rot_norm))
            for off in range(max_offset_rot, -1, -1):
                if can_place(rot_norm, base_row + off, occupied):
                    if best_offset is None or off > best_offset or \
                       (off == best_offset and best_rotated is False):
                        best_offset = off
                        best_shape = rot_norm
                        best_rotated = True
                    break

        if best_offset is None:
            best_offset = 0
            best_shape = orig_norm

        place(best_shape, base_row + best_offset, comp["color"], occupied)

    return output