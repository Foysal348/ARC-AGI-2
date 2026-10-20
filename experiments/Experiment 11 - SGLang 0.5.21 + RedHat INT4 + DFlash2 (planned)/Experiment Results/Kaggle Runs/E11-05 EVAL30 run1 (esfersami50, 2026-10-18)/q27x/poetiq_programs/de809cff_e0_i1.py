import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by detecting small enclosed "hole" regions of zeros.

    Rule inferred from the examples:
    1. Identify the two non-background colors present in the grid.
    2. Find 8-connected components of zeros.
    3. A zero component is considered a valid hole if every non-zero cell
       adjacent to the component (including diagonal adjacency) has the same
       color.
    4. For each valid hole:
       - mark the hole cells themselves with color 8;
       - color the 3x3 neighborhood around the hole with the other non-zero
         color.
    5. Overlapping operations are resolved by giving color 8 the highest priority,
       then the filled "other" color, then the original value.

    This matches the examples: isolated or small enclosed zero holes are replaced
    by a 3x3 block of the opposite color, with 8 placed at the hole location(s).
    """
    grid = np.array(grid, copy=True)
    if grid.size == 0:
        return grid

    h, w = grid.shape

    # Collect the two non-zero colors.
    nonzero = [int(c) for c in np.unique(grid) if int(c) != 0]
    if len(nonzero) < 2:
        return grid
    color_a, color_b = nonzero[0], nonzero[1]
    other = {color_a: color_b, color_b: color_a}

    # Label zero components using 8-connectivity.
    visited = np.zeros((h, w), dtype=bool)
    components = []

    from collections import deque

    def neighbors8(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                comp = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for nr, nc in neighbors8(cr, cc):
                        if not visited[nr, nc] and grid[nr, nc] == 0:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                            comp.append((nr, nc))
                components.append(comp)

    new_grid = grid.copy()

    for comp in components:
        # Gather all neighboring non-zero colors around the component.
        boundary_colors = set()
        for (r, c) in comp:
            for nr, nc in neighbors8(r, c):
                val = grid[nr, nc]
                if val != 0:
                    boundary_colors.add(int(val))

        # Valid hole: surrounded by exactly one non-zero color.
        if len(boundary_colors) != 1:
            continue

        surrounding = boundary_colors.pop()
        fill_color = other[surrounding]

        # Mark the component as 8.
        for r, c in comp:
            new_grid[r, c] = 8

        # Fill the 3x3 neighborhood around each component cell.
        for r, c in comp:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if new_grid[nr, nc] != 8:
                            new_grid[nr, nc] = fill_color

    return new_grid