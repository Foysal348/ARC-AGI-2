import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid.

    The rule inferred from the examples:
    - The non-background object in the top-left contains a 2x2 palette.
    - The four palette colors correspond to the four quadrants:
          top-left, top-right,
          bottom-left, bottom-right
    - The output is a 4x4 block in which each 2x2 quadrant is filled with
      the corresponding palette color.
    """
    if grid.size == 0:
        return np.zeros_like(grid)

    # Find non-zero colors.
    nonzero = np.argwhere(grid != 0)
    if len(nonzero) == 0:
        return grid.copy()

    # Locate the 2x2 palette: four distinct non-zero pixels forming a 2x2 block.
    palette = None
    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1] - 1):
            block = grid[r:r+2, c:c+2]
            vals = block.flatten().tolist()
            if all(v != 0 for v in vals):
                if len(set(vals)) == 4:
                    palette = block.copy()
                    break
        if palette is not None:
            break

    # Fallback: if no exact 2x2 palette is found, use the four most frequent
    # non-zero colors in reading order.
    if palette is None:
        colors = []
        for row in grid:
            for v in row:
                if v != 0 and v not in colors:
                    colors.append(int(v))
        if len(colors) >= 4:
            palette = np.array(colors[:4]).reshape(2, 2)
        else:
            return grid.copy()

    # Build output: each palette pixel becomes a 2x2 block.
    out = np.empty((palette.shape[0] * 2, palette.shape[1] * 2), dtype=grid.dtype)
    for i in range(palette.shape[0]):
        for j in range(palette.shape[1]):
            out[i*2:(i+1)*2, j*2:(j+1)*2] = palette[i, j]

    return out