import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the large ARC grid into the small "combined frame contents" grid.

    Rule inferred from the examples:
      1. Find all square hollow rectangular frames.
         A frame is a square whose four borders are entirely one non-zero color,
         and whose interior does not contain that same border color.
      2. Keep only frames whose interiors contain at least one zero.
      3. Choose the most common interior size among the valid frames.
         Ties are broken by preferring smaller interior size, then by stable order.
      4. XOR together the contents of all selected frames.
         - 0 and the frame border color are ignored.
         - Color 6 is tracked separately.
         - Every other non-zero color is tracked as a generic "other" mask.
      5. Build the output:
         - If the generic mask is odd, the cell is 2.
         - Else if the 6 mask is odd, the cell is 6.
         - Otherwise the cell is 0.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    colors = sorted({int(v) for v in grid.flat if int(v) != 0})

    def border_is_square(color: int, r1: int, r2: int, c1: int, c2: int) -> bool:
        """Return True if the rectangle border is fully color and is square."""
        if r2 - r1 != c2 - c1:
            return False

        # Top and bottom borders.
        if not np.all(grid[r1, c1:c2 + 1] == color):
            return False
        if not np.all(grid[r2, c1:c2 + 1] == color):
            return False

        # Left and right borders.
        if not np.all(grid[r1:r2 + 1, c1] == color):
            return False
        if not np.all(grid[r1:r2 + 1, c2] == color):
            return False

        # Interior must not contain the border color.
        interior = grid[r1 + 1:r2, c1 + 1:c2]
        return not np.any(interior == color)

    frames = []

    for color in colors:
        mask = (grid == color).astype(np.int8)

        # For each row, find horizontal runs of this color.
        for r1 in range(H):
            row = mask[r1]
            c = 0
            while c < W:
                if not row[c]:
                    c += 1
                    continue

                c2 = c
                while c2 + 1 < W and row[c2 + 1]:
                    c2 += 1

                width = c2 - c + 1
                if width >= 3:
                    # A valid square frame has top row length == height.
                    r2 = r1 + width - 1
                    if r2 < H:
                        if border_is_square(color, r1, r2, c, c2):
                            interior = grid[r1 + 1:r2, c + 1:c2]
                            if np.any(interior == 0):
                                frames.append((color, r1, r2, c, c2))

                c = c2 + 1

    if not frames:
        return np.zeros_like(grid)

    # Group frames by interior size.
    groups = {}
    for color, r1, r2, c1, c2 in frames:
        ih, iw = r2 - r1 - 1, c2 - c1 - 1
        if ih <= 0 or iw <= 0:
            continue
        key = (ih, iw)
        groups.setdefault(key, []).append((color, r1, r2, c1, c2))

    # Choose the best interior size.
    # The examples use the common small square interior size.
    best_key = min(
        groups.items(),
        key=lambda kv: (-len(kv[1]), kv[0][0] + kv[0][1])
    )[0]

    ih, iw = best_key
    selected = groups[best_key]

    mask_6 = np.zeros((ih, iw), dtype=bool)
    mask_other = np.zeros((ih, iw), dtype=bool)

    for color, r1, r2, c1, c2 in selected:
        interior = grid[r1 + 1:r2, c1 + 1:c2]
        for i in range(ih):
            for j in range(iw):
                v = int(interior[i, j])
                if v == 0 or v == color:
                    continue

                if v == 6:
                    mask_6[i, j] = not mask_6[i, j]
                else:
                    mask_other[i, j] = not mask_other[i, j]

    out = np.zeros((ih, iw), dtype=grid.dtype)
    out[mask_other] = 2
    out[mask_6 & ~mask_other] = 6

    return out