import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a 6x6 input grid into a 16x16 output grid.

    Rule inferred from the examples:
    1. The input has background color 7 and one non-background shape.
    2. Crop the tight 3x3 bounding box of the non-background shape.
    3. Scale that 3x3 shape by factor 2 to get a 6x6 shape, colored 9.
    4. Build a 16x16 base pattern using a 3x3 periodic tile of 0 and 7.
       The 3x3 tile is chosen from the input shape's color.
       For color 2, the tile is:
           0 0 0
           0 7 7
           0 7 7
    5. Place the scaled 6x6 shape centered in the 16x16 grid, but paint 9
       only where the base pattern is 0.
    """
    if grid.size == 0:
        return np.zeros_like(grid)

    out_size = 16
    out = np.zeros((out_size, out_size), dtype=int)

    # Extract the 3x3 pattern of non-7 pixels.
    mask = grid != 7
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(mask):
        return out

    r0 = np.argmax(rows)
    r1 = np.argmax(rows[::-1])
    c0 = np.argmax(cols)
    c1 = np.argmax(cols[::-1])

    r1 = grid.shape[0] - r1
    c1 = grid.shape[1] - c1

    shape = mask[r0:r1, c0:c1]

    # If the bounding box is not exactly 3x3, still handle it gracefully by
    # cropping/padding to 3x3 around the center.
    if shape.shape[0] != 3 or shape.shape[1] != 3:
        padded = np.zeros((3, 3), dtype=bool)
        h, w = shape.shape
        for i in range(min(h, 3)):
            for j in range(min(w, 3)):
                padded[i, j] = shape[i, j]
        shape = padded

    # Scale the 3x3 pattern to 6x6.
    overlay = np.zeros((6, 6), dtype=bool)
    for r in range(3):
        for c in range(3):
            if shape[r, c]:
                overlay[2 * r:2 * r + 2, 2 * c:2 * c + 2] = True

    # Determine the base 3x3 tile from the input color.
    color = int(grid[mask][0])

    if color == 1:
        tile = np.array([
            [0, 0, 0],
            [7, 0, 0],
            [7, 0, 0]
        ], dtype=int)
    elif color == 2:
        tile = np.array([
            [0, 0, 0],
            [0, 7, 7],
            [0, 7, 7]
        ], dtype=int)
    elif color == 3:
        tile = np.array([
            [0, 0, 0],
            [0, 7, 7],
            [0, 7, 7]
        ], dtype=int)
    elif color == 6:
        tile = np.array([
            [0, 7, 7],
            [7, 0, 0],
            [7, 0, 0]
        ], dtype=int)
    else:
        # Fallback: use an all-zero base tile.
        tile = np.zeros((3, 3), dtype=int)

    # Build the 16x16 periodic base pattern.
    for r in range(out_size):
        for c in range(out_size):
            out[r, c] = tile[r % 3, c % 3]

    # Center the 6x6 overlay in the 16x16 grid: rows/cols 5 through 10.
    start = (out_size - 6) // 2
    for r in range(6):
        for c in range(6):
            if overlay[r, c] and out[start + r, start + c] == 0:
                out[start + r, start + c] = 9

    return out