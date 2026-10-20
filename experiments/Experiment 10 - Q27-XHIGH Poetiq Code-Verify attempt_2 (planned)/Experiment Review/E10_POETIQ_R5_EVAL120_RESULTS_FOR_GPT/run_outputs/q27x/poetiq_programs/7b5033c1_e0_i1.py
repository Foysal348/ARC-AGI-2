import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extract the colored path(s) from the background and return them as a
    one-column grid.

    The background is identified as the uniform corner color, or the most
    frequent color if the corners differ. All other cells are treated as
    4-connected components. Each component is traversed from its topmost /
    leftmost endpoint to the other endpoint if it is a simple path. For
    unexpected non-path components, cells are output in row-major order.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return np.empty((0, 1), dtype=grid.dtype)
    if grid.ndim == 1:
        grid = grid.reshape(1, -1)

    # Identify background color.
    corners = np.array([grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]])
    if len(np.unique(corners)) == 1:
        background = corners[0]
    else:
        values, counts = np.unique(grid, return_counts=True)
        background = values[np.argmax(counts)]

    mask = grid != background
    H, W = grid.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Find connected non-background components.
    visited = np.zeros((H, W), dtype=bool)
    components = []

    for r in range(H):
        for c in range(W):
            if mask[r, c] and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                cells = []

                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))

                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                components.append(cells)

    if not components:
        return np.empty((0, 1), dtype=grid.dtype)

    # Process components in reading order.
    components.sort(key=lambda comp: min(comp))

    def component_values(comp_cells):
        cell_set = set(comp_cells)

        if len(cell_set) == 1:
            r, c = next(iter(cell_set))
            return [int(grid[r, c])]

        # Build adjacency for this component.
        neighbors = {}
        for r, c in cell_set:
            nbrs = []
            if r > 0 and (r - 1, c) in cell_set:
                nbrs.append((r - 1, c))
            if r + 1 < H and (r + 1, c) in cell_set:
                nbrs.append((r + 1, c))
            if c > 0 and (r, c - 1) in cell_set:
                nbrs.append((r, c - 1))
            if c + 1 < W and (r, c + 1) in cell_set:
                nbrs.append((r, c + 1))
            neighbors[(r, c)] = nbrs

        degrees = {cell: len(nbrs) for cell, nbrs in neighbors.items()}
        endpoints = [cell for cell, deg in degrees.items() if deg == 1]
        max_deg = max(degrees.values()) if degrees else 0

        # Traverse simple paths and cycles in path order.
        if (len(endpoints) == 2 and max_deg <= 2) or (len(endpoints) == 0 and max_deg == 2):
            start = min(endpoints) if endpoints else min(cell_set)
            seq = []
            seen = set()
            prev = None
            curr = start

            while curr is not None:
                if curr in seen:
                    break

                seen.add(curr)
                seq.append(int(grid[curr[0], curr[1]]))

                nexts = [n for n in neighbors[curr] if n != prev and n not in seen]
                if not nexts:
                    break

                curr_next = min(nexts)
                prev, curr = curr, curr_next

            if len(seen) == len(cell_set):
                return seq

        # Fallback for unexpected shapes: row-major order.
        return [int(grid[r, c]) for r, c in sorted(comp_cells)]

    result = []
    for comp in components:
        result.extend(component_values(comp))

    if not result:
        return np.empty((0, 1), dtype=grid.dtype)

    return np.array(result, dtype=grid.dtype).reshape(-1, 1)