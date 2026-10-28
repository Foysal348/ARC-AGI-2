import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Detect colored square tiles, separate solid tiles from hollow tiles,
    and arrange them into a two-column compact grid:
      - left column: hollow tiles, sorted by vertical position
      - right column: solid tiles, sorted by vertical position
    Each tile is copied unchanged. Empty slots are zeros.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2:
        return np.zeros_like(grid)

    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    tiles = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 and not visited[r, c]:
                color = grid[r, c]
                stack = [(r, c)]
                visited[r, c] = True
                cells = []

                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))

                    for dr, dc in directions:
                        nr, nc = rr + dr, cc + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and not visited[nr, nc]
                            and grid[nr, nc] == color
                        ):
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                r0 = min(cell[0] for cell in cells)
                r1 = max(cell[0] for cell in cells)
                c0 = min(cell[1] for cell in cells)
                c1 = max(cell[1] for cell in cells)

                patch = grid[r0:r1 + 1, c0:c1 + 1].copy()

                # A tile is solid if its bounding box is completely filled.
                is_solid = bool(np.all(patch == color))

                tiles.append((r0, c0, patch, is_solid))

    if not tiles:
        return np.zeros_like(grid)

    hollows = [tile for tile in tiles if not tile[3]]
    solids = [tile for tile in tiles if tile[3]]

    hollows.sort(key=lambda tile: (tile[0], tile[1]))
    solids.sort(key=lambda tile: (tile[0], tile[1]))

    block_h = max(tile[2].shape[0] for tile in tiles)
    block_w = max(tile[2].shape[1] for tile in tiles)
    rows = max(len(hollows), len(solids))

    out = np.zeros((rows * block_h, 2 * block_w), dtype=grid.dtype)

    for i, (_, _, patch, _) in enumerate(hollows):
        ph, pw = patch.shape
        out[i * block_h:i * block_h + ph, 0:pw] = patch

    for i, (_, _, patch, _) in enumerate(solids):
        ph, pw = patch.shape
        out[i * block_h:i * block_h + ph, block_w:block_w + pw] = patch

    return out