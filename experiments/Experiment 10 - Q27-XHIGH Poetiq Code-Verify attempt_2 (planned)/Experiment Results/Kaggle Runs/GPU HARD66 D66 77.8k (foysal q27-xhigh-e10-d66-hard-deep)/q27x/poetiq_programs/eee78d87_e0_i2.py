import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the 6x6 input into the 16x16 output.

    Rule inferred from examples:
    1. The input background is color 7.  The small non-background shape is
       extracted as a 3x3 binary pattern.
    2. The foreground color determines a 3x3 background tile of 0/7:
          color 1 -> left lower column
          color 2 -> top right row
          color 3 -> bottom right 2x2 block
          color 6 -> XOR of the previous two (left lower + top right)
       This tile is repeated across the whole 16x16 grid.
    3. In the central 6x6 region, stamp the extracted 3x3 pattern tiled
       twice horizontally and twice vertically, using color 9 for pattern
       cells.  The top-left cell of the stamped 3x3 pattern is ignored.
       Where the stamp is 0, keep the underlying 0/7 background tile.
    """
    grid = np.asarray(grid, dtype=int)

    # Find non-background cells.
    fg_mask = grid != 7
    if not fg_mask.any():
        # Fallback: all-zero 16x16 grid.
        return np.zeros((16, 16), dtype=int)

    ys, xs = np.nonzero(fg_mask)
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())

    # Foreground color: any non-7 color in the pattern.
    fg_color = int(grid[ys[0], xs[0]])

    # Extract 3x3 binary pattern.
    pattern = np.zeros((3, 3), dtype=bool)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    if h <= 3 and w <= 3:
        pattern[:h, :w] = fg_mask[r0:r1 + 1, c0:c1 + 1]
    else:
        # Fallback: use the densest 3x3 window.
        best_r, best_c, best_cnt = 0, 0, -1
        for r in range(grid.shape[0] - 2):
            for c in range(grid.shape[1] - 2):
                cnt = int(fg_mask[r:r + 3, c:c + 3].sum())
                if cnt > best_cnt:
                    best_r, best_c, best_cnt = r, c
        if best_cnt > 0:
            pattern = fg_mask[best_r:best_r + 3, best_c:best_c + 3].copy()

    # Ignore the top-left cell of the central stamp.
    pattern[0, 0] = False

    # Choose the 3x3 background tile based on foreground color.
    tile = np.zeros((3, 3), dtype=int)
    if fg_color == 1:
        # Left lower column.
        tile[1, 0] = 7
        tile[2, 0] = 7
    elif fg_color == 2:
        # Top right row.
        tile[0, 1] = 7
        tile[0, 2] = 7
    elif fg_color == 3:
        # Bottom right 2x2 block.
        tile[1, 1] = 7
        tile[1, 2] = 7
        tile[2, 1] = 7
        tile[2, 2] = 7
    elif fg_color == 6:
        # Left lower column + top right row.
        tile[0, 1] = 7
        tile[0, 2] = 7
        tile[1, 0] = 7
        tile[2, 0] = 7
    else:
        # Graceful fallback: use bottom-right 2x2 if unknown color.
        tile[1, 1] = 7
        tile[1, 2] = 7
        tile[2, 1] = 7
        tile[2, 2] = 7

    # Repeat tile to make 16x16 background.
    out = np.tile(tile, (16 // 3 + 1, 16 // 3 + 1))[:16, :16].astype(int)

    # Central 6x6 stamp: tile the 3x3 pattern 2x2.
    stamp = np.tile(pattern, (2, 2))
    out[5:11, 5:11] = np.where(stamp, 9, out[5:11, 5:11])

    return out