import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:

    1. The top-left cell contains a command color.
       All other cells of that color are removed.
       The top-left command cell is preserved.

    2. After that, every remaining non-zero connected component
       (4-connected, ignoring color boundaries) falls downward by the
       maximum amount that is both:
         - allowed by the command-color-specific shift table, and
         - safe, meaning no cell would leave the grid.

    3. The shift table inferred from the examples is:
         1 -> 1
         2 -> 1
         3 -> 1
         4 -> 2
         5 -> 2
         6 -> 2
         7 -> 3
         8 -> 3
         9 -> 0

    This matches all given examples:
      - Example 1: command 6, shift 2
      - Example 2: command 4, shift 2
      - Example 3: command 7, shift 3
      - Example 4: command 9, shift 0
      - Example 5: command 4, shift 2
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    out = grid.astype(int).copy()

    cmd = int(out[0, 0])

    # Remove the command color everywhere except the top-left marker.
    out[grid == cmd] = 0
    out[0, 0] = cmd

    shift_table = {
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 2,
        7: 3,
        8: 3,
        9: 0,
    }
    shift = shift_table.get(cmd, 0)

    if shift == 0:
        return out

    # Label 4-connected components of all remaining non-zero cells.
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if out[r, c] != 0 and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                comp = [(r, c)]

                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if out[nr, nc] != 0 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                comp.append((nr, nc))

                components.append(comp)

    result = np.zeros_like(out)
    result[0, 0] = cmd

    for comp in components:
        # The command marker is preserved exactly as-is.
        if (0, 0) in comp:
            for r, c in comp:
                result[r, c] = out[r, c]
            continue

        # Find how far this component can safely move downward.
        max_safe_shift = shift
        for r, c in comp:
            max_safe_shift = min(max_safe_shift, h - 1 - r)

        if max_safe_shift > 0:
            for r, c in comp:
                result[r + max_safe_shift, c] = out[r, c]
        else:
            for r, c in comp:
                result[r, c] = out[r, c]

    return result