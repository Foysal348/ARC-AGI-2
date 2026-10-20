import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule inferred from the examples:

    1. The grid contains background (0) and exactly two foreground colors.
    2. Remove tiny foreground specks: any 4-connected component of a
       foreground color that consists of only one pixel is erased.
    3. Find "hole centers": 0 cells that have at least three orthogonal
       neighbors (up/down/left/right) with the same non-zero color.  These
       are the cells that become 8 in the output.
    4. For each hole center, paint its 3x3 neighborhood with the opposite
       foreground color.  The center itself becomes 8.
    5. If multiple operations touch the same cell, 8 has highest priority.
       For conflicting non-8 fills, the cell is left unchanged.

    This reproduces the examples: isolated noise pixels vanish, and small
    zero holes surrounded mostly by one color become 3x3 blocks of the other
    color, marked with 8 at the hole center.
    """
    grid = np.array(grid, copy=True)
    if grid.size == 0:
        return grid

    h, w = grid.shape

    # Identify the two foreground colors.
    colors = [int(c) for c in np.unique(grid) if int(c) != 0]
    if len(colors) != 2:
        return grid
    color_a, color_b = colors
    other_color = {color_a: color_b, color_b: color_a}

    # -------------------------------------------------------------------------
    # 1) Remove isolated foreground pixels (4-connected components of size 1).
    # -------------------------------------------------------------------------
    erased = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val == 0 or visited[r, c]:
                continue

            # 4-connected component of this color.
            comp = [(r, c)]
            visited[r, c] = True
            q = deque([(r, c)])

            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == val
                    ):
                        visited[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))

            if len(comp) == 1:
                erased[r, c] = True

    for r, c in np.argwhere(erased):
        grid[r, c] = 0

    # -------------------------------------------------------------------------
    # 2) Find hole centers:
    #    zero cells with at least 3 orthogonal neighbors of one foreground color.
    # -------------------------------------------------------------------------
    centers = []
    center_fill = {}

    for r in range(h):
        if grid[r, c] != 0:
            continue

        counts = {color_a: 0, color_b: 0}
        nonzero_cardinal = 0

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                val = int(grid[nr, nc])
                if val in counts:
                    counts[val] += 1
                    nonzero_cardinal += 1

        # Need at least three cardinal neighbors, all of the same foreground color.
        if nonzero_cardinal >= 3:
            chosen = None
            if counts[color_a] >= 3 and counts[color_b] == 0:
                chosen = color_a
            elif counts[color_b] >= 3 and counts[color_a] == 0:
                chosen = color_b

            if chosen is not None:
                centers.append((r, c))
                center_fill[(r, c)] = other_color[chosen]

    if not centers:
        return grid

    # -------------------------------------------------------------------------
    # 3) Apply 3x3 fills and mark centers with 8.
    # -------------------------------------------------------------------------
    fill_vote = {}

    for (r, c), fill in center_fill.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    fill_vote.setdefault((nr, nc), set()).add(fill)

    new_grid = grid.copy()

    # Non-8 fills: if conflicting fills overlap, leave the original value.
    for (r, c), votes in fill_vote.items():
        if len(votes) == 1:
            new_grid[r, c] = next(iter(votes))

    # Centers have priority.
    for r, c in centers:
        new_grid[r, c] = 8

    return new_grid