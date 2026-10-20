import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule inferred from the examples:

    - The grid uses background 0 and exactly two foreground colors.
    - Isolated single-pixel foreground specks are removed.
    - Every remaining background cell is tested locally.  If at least 5 of its
      8 neighbouring cells are the same non-zero color, it is treated as a
      "hole center".
    - Each hole center is marked with 8.
    - The 8 neighbours around each hole center are painted with the opposite
      foreground color.
    - If multiple fills conflict, the value 8 has priority; otherwise the first
      consistent fill is kept and contradictory fills leave the cell unchanged.

    This matches the provided examples: small zero holes and small local
    indentations are inverted into 3x3 blocks of the other foreground color,
    with 8 marking the selected center.
    """
    grid = np.array(grid, copy=True)
    if grid.size == 0:
        return grid

    h, w = grid.shape

    # ------------------------------------------------------------------
    # Determine the two non-background colors.
    # ------------------------------------------------------------------
    colors = [int(c) for c in np.unique(grid) if int(c) != 0]
    if len(colors) < 2:
        return grid

    # Use the first two colors if there are extra colors; in the examples
    # exactly two foreground colors appear.
    color_a, color_b = colors[0], colors[1]
    other = {color_a: color_b, color_b: color_a}
    fg = {color_a, color_b}

    # ------------------------------------------------------------------
    # Remove isolated 4-connected foreground pixels.
    # ------------------------------------------------------------------
    visited = np.zeros((h, w), dtype=bool)
    erase = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            comp = [(r, c)]

            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == int(grid[r, c]):
                        visited[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))

            if len(comp) == 1:
                erase[r, c] = True

    for r, c in np.argwhere(erase):
        grid[r, c] = 0

    # ------------------------------------------------------------------
    # Find local hole centers:
    #   zero cells with at least five of the eight neighbours of the same
    #   foreground color.
    # ------------------------------------------------------------------
    centers = []
    fill_for_center = {}

    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0:
                continue

            count_a = 0
            count_b = 0

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        val = int(grid[nr, nc])
                        if val == color_a:
                            count_a += 1
                        elif val == color_b:
                            count_b += 1

            if count_a >= 5 and count_a > count_b:
                centers.append((r, c))
                fill_for_center[(r, c)] = color_b
            elif count_b >= 5 and count_b > count_a:
                centers.append((r, c))
                fill_for_center[(r, c)] = color_a

    if not centers:
        return grid

    # ------------------------------------------------------------------
    # Apply 3x3 fills.  8 is highest priority.  Conflicting non-8 fills are
    # resolved conservatively: if two different fills overlap, leave original.
    # ------------------------------------------------------------------
    votes = {}
    for (r, c), fill in fill_for_center.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    votes.setdefault((nr, nc), set()).add(fill)

    new_grid = grid.copy()

    for (r, c), vals in votes.items():
        if len(vals) == 1:
            new_grid[r, c] = next(iter(vals))

    for r, c in centers:
        new_grid[r, c] = 8

    return new_grid