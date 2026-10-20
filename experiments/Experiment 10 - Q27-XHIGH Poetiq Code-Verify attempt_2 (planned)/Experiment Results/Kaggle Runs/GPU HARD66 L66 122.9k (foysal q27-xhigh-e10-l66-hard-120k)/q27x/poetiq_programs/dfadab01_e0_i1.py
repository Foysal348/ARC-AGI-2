import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Expand certain single-pixel seeds into 4x4 stamps.

    Seed colors and their stamps:
      2 -> color 4, full 4x4 border
      3 -> color 1, edge-midpoint diamond
      5 -> color 6, two diagonal 2x2 blocks
      8 -> color 7, corner/inner diamond

    A seed is used only if the 4x4 square with the seed as its top-left
    corner is fully inside the grid and contains no other seed pixel.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2:
        return np.zeros_like(grid, dtype=int)

    patterns = {
        2: (4, np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ], dtype=int)),

        3: (1, np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ], dtype=int)),

        5: (6, np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ], dtype=int)),

        8: (7, np.array([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ], dtype=int)),
    }

    h, w = grid.shape
    out = np.zeros((h, w), dtype=int)

    seed_mask = np.isin(grid, list(patterns.keys()))

    for r in range(h):
        for c in range(w):
            if not seed_mask[r, c]:
                continue

            # The 4x4 stamp must fit completely inside the grid.
            if r + 3 >= h or c + 3 >= w:
                continue

            # Ignore seeds whose 4x4 footprint contains another seed.
            if seed_mask[r:r + 4, c:c + 4].sum() != 1:
                continue

            color, pattern = patterns[grid[r, c]]
            block = out[r:r + 4, c:c + 4]
            block[pattern.astype(bool)] = color

    return out