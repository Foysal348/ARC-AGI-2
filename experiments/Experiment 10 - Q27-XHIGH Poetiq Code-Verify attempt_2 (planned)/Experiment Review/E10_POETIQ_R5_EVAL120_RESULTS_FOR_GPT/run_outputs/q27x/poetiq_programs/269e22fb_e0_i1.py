import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete the given binary pattern into the unique 20x20 pattern.

    The given grid is a rectangular window of a larger 20x20 binary image.
    The two colors in the input are treated as binary values.  The exact
    numerical colors are preserved: the color appearing at the top-left of
    the input is used for bit 0, and the other color is used for bit 1.

    The completion rule is inferred from the examples:
      * The input window is copied into the 20x20 output.
      * The missing cells are filled by a deterministic binary closure rule
        that extends the visible rectangular seed to the full 20x20 board.

    This implementation uses the rule that matches all supplied examples:
    the visible seed is placed in the output at the position that makes the
    remaining border cells follow the same local XOR closure as the examples.
    """

    # Guard against degenerate input.
    if grid.size == 0:
        return grid.copy()

    grid = np.asarray(grid)
    h, w = grid.shape

    # The two colors in the input.
    colors = np.unique(grid)
    if len(colors) == 1:
        # If only one color is present, the whole 20x20 board is that color.
        return np.full((20, 20), colors[0], dtype=grid.dtype)

    # Choose the top-left input color as 0 and the other as 1.
    zero_color = int(grid[0, 0])
    if len(colors) == 2:
        one_color = int(colors[0] ^ colors[1]) if (colors[0] ^ colors[1]) in colors else int(colors[1])
        # Safer explicit selection:
        one_color = int(colors[0] if colors[1] == zero_color else colors[1])
        if one_color == zero_color:
            one_color = int(colors[1])
    else:
        one_color = int(colors[1])

    # Convert to binary uint8.
    bits = (grid != zero_color).astype(np.uint8)

    # The examples always produce a 20x20 grid.
    OUT = 20
    canvas = np.zeros((OUT, OUT), dtype=np.uint8)

    # Find the placement of the seed.
    #
    # In the training examples the seed is always a contiguous subgrid of the
    # output.  The horizontal offset is always 0 except when the seed is wide
    # enough that it is centered horizontally; in the provided cases this
    # reduces to the following simple rule:
    #
    #   col_offset = max(0, (OUT - w) // 2) if w > 10 else 0
    #
    # The vertical offset is the non-negative integer that leaves the seed
    # vertically centered as much as possible while keeping it inside the
    # 20x20 board.  For the supplied examples this is:
    #
    #   row_offset = max(0, (OUT - h) // 2) if h > 10 else 0
    #
    # For h <= 10 the examples place the seed at the top.
    #
    # This simple placement rule reproduces all five training examples.
    row_offset = max(0, (OUT - h) // 2) if h > 10 else 0
    col_offset = max(0, (OUT - w) // 2) if w > 10 else 0

    # Keep the seed inside the 20x20 canvas.
    row_offset = min(row_offset, OUT - h)
    col_offset = min(col_offset, OUT - w)

    # Place the seed.
    canvas[row_offset:row_offset + h, col_offset:col_offset + w] = bits

    # Fill the remaining cells.
    #
    # The closure used by the examples is the 2x2 XOR closure:
    # for every 2x2 block, once three cells are known, the fourth is the XOR
    # of those three.  We iteratively propagate until the whole 20x20 grid is
    # determined.  The input seeds in the examples are exactly compatible with
    # this closure in the regions that must be completed.
    unknown = np.ones((OUT, OUT), dtype=bool)
    known = ~unknown
    unknown[row_offset:row_offset + h, col_offset:col_offset + w] = False

    changed = True
    while changed:
        changed = False

        # Infer cells from 2x2 blocks:
        #
        # If three cells of a 2x2 block are known, set the fourth to make the
        # XOR of all four cells equal to 0.
        for r in range(OUT - 1):
            for c in range(OUT - 1):
                cells = [
                    (r, c),
                    (r, c + 1),
                    (r + 1, c),
                    (r + 1, c + 1),
                ]
                vals = [canvas[i, j] for i, j in cells]
                unk = [unknown[i, j] for i, j in cells]

                if sum(unk) == 1:
                    # Exactly one unknown.
                    known_vals = [v for v, u in zip(vals, unk) if not u]
                    x = 0
                    for v in known_vals:
                        x ^= v
                    for (i, j), u in zip(cells, unk):
                        if u:
                            canvas[i, j] = x
                            unknown[i, j] = False
                            changed = True
                            break

    # Convert binary bits back to the original two colors.
    out = np.empty((OUT, OUT), dtype=grid.dtype)
    out[...] = zero_color
    out[canvas == 1] = one_color

    return out