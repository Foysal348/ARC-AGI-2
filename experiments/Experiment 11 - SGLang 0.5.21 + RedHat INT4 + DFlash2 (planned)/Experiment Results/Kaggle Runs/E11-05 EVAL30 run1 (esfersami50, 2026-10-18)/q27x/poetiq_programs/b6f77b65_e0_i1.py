import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    The top-left value selects a color to remove.  All remaining non-zero
    connected components then move downward by a fixed number of rows chosen
    from the command color.

    The examples show the command behavior:
        2, 3 -> shift down 1
        4    -> shift down 2
        5    -> shift down 3
        6, 7 -> shift down 4
        8    -> shift down 5
        9    -> no change

    The implementation:
      1. Copy the grid.
      2. Read the command color at (0, 0).
      3. Remove all cells of that color except the command cell itself.
      4. Find connected components of the remaining non-zero cells,
         including the command cell as a separate component.
      5. Move every component except the command component downward by the
         command-specific amount, clipping cells that fall outside the grid.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    out = grid.astype(int).copy()

    cmd = int(grid[0, 0])

    # Remove the command color everywhere except the command marker itself.
    out[grid == cmd] = 0
    out[0, 0] = cmd

    # Determine the downward shift from the command color.
    shifts = {
        1: 0,
        2: 1,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 4,
        8: 5,
        9: 0,
    }
    shift = shifts.get(cmd, 0)

    if shift == 0:
        return out

    # Label connected components of non-zero cells in the copied grid.
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
                                q.append((nr, nc))
                                comp.append((nr, nc))
                components.append(comp)

    # Clear all non-zero cells, then place shifted components back.
    result = np.zeros_like(out)
    result[0, 0] = cmd  # preserve the command marker

    for comp in components:
        # Do not move the command marker itself.
        if (0, 0) in comp:
            continue

        for r, c in comp:
            nr = r + shift
            if 0 <= nr < h and 0 <= c < w:
                result[nr, c] = out[r, c]

    return result