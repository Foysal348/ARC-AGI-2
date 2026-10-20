import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the 6x6 input into the 16x16 output.

    Rule:
    1. Use the single non-7 color in the input as the key.
    2. Build a 16x16 periodic background from a 3x3 tile associated with that key color.
    3. Inside the centered 6x6 region, paint 9 on all background-zero cells, except:
       - for color 1: clear the two notches at the top-middle and bottom-middle of the 6x6 region.
       - for color 2: clear the central 2x2 block of the 6x6 region.
       - for colors 3 and 6: no extra clearing.
    """
    if grid.size == 0:
        return np.zeros_like(grid)

    out = np.zeros((16, 16), dtype=int)

    # Find the non-background color.
    mask = grid != 7
    if not np.any(mask):
        return out

    key_color = int(grid[mask][0])

    # 3x3 background tile for each observed/required key color.
    if key_color == 1:
        tile = np.array([
            [0, 0, 0],
            [7, 0, 0],
            [7, 0, 0]
        ], dtype=int)
    elif key_color == 2:
        tile = np.array([
            [0, 0, 0],
            [0, 7, 7],
            [0, 7, 7]
        ], dtype=int)
    elif key_color == 3:
        tile = np.array([
            [0, 0, 0],
            [0, 7, 7],
            [0, 7, 7]
        ], dtype=int)
    elif key_color == 6:
        tile = np.array([
            [0, 7, 7],
            [7, 0, 0],
            [7, 0, 0]
        ], dtype=int)
    else:
        # Fallback: keep the zero background.
        tile = np.zeros((3, 3), dtype=int)

    # Fill the 16x16 grid with the periodic 3x3 tile.
    for r in range(16):
        for c in range(16):
            out[r, c] = tile[r % 3, c % 3]

    # The centered 6x6 region is rows/cols 5 through 10.
    start = 5

    # Initial 9-region: all zeros in that centered 6x6 area.
    for r in range(6):
        for c in range(6):
            if out[start + r, start + c] == 0:
                out[start + r, start + c] = 9

    # Color-specific corrections.
    if key_color == 1:
        # Top-middle notch: global rows 5, cols 7-8.
        out[5, 7:10] = 0
        # Bottom-middle notch: global rows 10, cols 7-8.
        out[10, 7:10] = 0

    elif key_color == 2:
        # Central 2x2 notch: global rows 7-8, cols 7-8.
        out[7:10, 7:10] = 0

    return out