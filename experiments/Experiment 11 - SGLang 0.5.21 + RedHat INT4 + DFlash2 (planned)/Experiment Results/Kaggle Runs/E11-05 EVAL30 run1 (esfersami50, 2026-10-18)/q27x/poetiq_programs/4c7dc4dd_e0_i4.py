import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by extracting all relevant square "frame" interiors
    and combining them with bitwise XOR.

    Inference from examples:
    ----------------------
    - The large input contains several square hollow frames.
    - A frame is a square whose border is made entirely of one non-zero color.
    - The interior must not contain the border color, and must contain at least
      one zero.
    - Among all such frames, use the most common interior size.
      Ties are broken by preferring the smaller interior.
    - For all selected frames, XOR their interiors together.
    - In the resulting parity grid:
        * color 6 is kept separately and output as 6 where it appears an odd
          number of times and no other non-zero color appears there;
        * every other non-zero, non-frame color is treated as a generic
          foreground color and output as 2.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    if H < 3 or W < 3:
        return np.zeros_like(grid)

    colors = sorted({int(v) for v in grid.flat if int(v) != 0})

    # Prefix sum for zero counts inside candidate interiors.
    zero_prefix = np.zeros((H + 1, W + 1), dtype=np.int32)
    zero_prefix[1:, 1:] = (grid == 0).astype(np.int32).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(prefix, r1, c1, r2, c2):
        """Inclusive rectangle sum [r1..r2] x [c1..c2]."""
        return int(
            prefix[r2 + 1, c2 + 1]
            - prefix[r1, c2 + 1]
            - prefix[r2 + 1, c1]
            + prefix[r1, c1]
        )

    frames = []

    # Detect all square hollow frames.
    for color in colors:
        mask = (grid == color).astype(np.int32)
        prefix = np.zeros((H + 1, W + 1), dtype=np.int32)
        prefix[1:, 1:] = mask.cumsum(axis=0).cumsum(axis=1)

        max_side = min(H, W)

        for side in range(3, max_side + 1):
            inner = side - 2
            if inner <= 0:
                continue

            # Number of cells on the border of a side x side square.
            border_area = 2 * side + 2 * (side - 2)

            for r1 in range(H - side + 1):
                r2 = r1 + side - 1

                # Quick row filter.
                if rect_sum(prefix, r1, 0, r1, W - 1) == 0:
                    continue
                if rect_sum(prefix, r2, 0, r2, W - 1) == 0:
                    continue

                for c1 in range(W - side + 1):
                    c2 = c1 + side - 1

                    # Top and bottom rows must be fully the frame color.
                    if rect_sum(prefix, r1, c1, r1, c2) != side:
                        continue
                    if rect_sum(prefix, r2, c1, r2, c2) != side:
                        continue

                    # Left and right columns must be fully the frame color.
                    if rect_sum(prefix, r1, c1, r2, c1) != side:
                        continue
                    if rect_sum(prefix, r1, c2, r2, c2) != side:
                        continue

                    # Full border consistency check.
                    border_count = (
                        rect_sum(prefix, r1, c1, r1, c2)
                        + rect_sum(prefix, r2, c1, r2, c2)
                        + rect_sum(prefix, r1 + 1, c1, r2 - 1, c1)
                        + rect_sum(prefix, r1 + 1, c2, r2 - 1, c2)
                    )
                    if border_count != border_area:
                        continue

                    # Interior must not contain the border color.
                    if rect_sum(prefix, r1 + 1, c1 + 1, r2 - 1, c2 - 1) != 0:
                        continue

                    # Interior must contain at least one zero.
                    if rect_sum(zero_prefix, r1 + 1, c1 + 1, r2 - 1, c2 - 1) == 0:
                        continue

                    frames.append((color, r1, c1, side))

    if not frames:
        return np.zeros_like(grid)

    # Group frames by interior dimensions.
    groups = {}
    for color, r1, c1, side in frames:
        inner_h = side - 2
        inner_w = side - 2
        if inner_h <= 0 or inner_w <= 0:
            continue

        key = (inner_h, inner_w)
        groups.setdefault(key, []).append((color, r1, c1, side))

    if not groups:
        return np.zeros_like(grid)

    # Choose the most common interior size.
    # Tie-break: smaller area first, then smaller height+width.
    best_key = min(
        groups.items(),
        key=lambda kv: (-len(kv[1]), kv[0][0] * kv[0][1], kv[0][0] + kv[0][1])
    )[0]

    ih, iw = best_key
    selected_frames = groups[best_key]

    # XOR masks:
    #   mask_6: parity of color 6
    #   mask_other: parity of all other non-zero, non-frame colors
    mask_6 = np.zeros((ih, iw), dtype=bool)
    mask_other = np.zeros((ih, iw), dtype=bool)

    for color, r1, c1, side in selected_frames:
        interior = grid[r1 + 1:r1 + 1 + ih, c1 + 1:c1 + 1 + iw]

        for i in range(ih):
            for j in range(iw):
                v = int(interior[i, j])

                # Ignore background and the frame's own border color.
                if v == 0 or v == color:
                    continue

                if v == 6:
                    mask_6[i, j] = not mask_6[i, j]
                else:
                    mask_other[i, j] = not mask_other[i, j]

    # Build output:
    #   generic foreground has priority and is output as 2,
    #   otherwise isolated 6 parity is output as 6,
    #   otherwise 0.
    out = np.zeros((ih, iw), dtype=grid.dtype)
    out[mask_other] = 2
    out[mask_6 & ~mask_other] = 6

    return out