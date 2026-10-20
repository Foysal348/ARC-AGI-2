import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the large input grid into the small output grid.

    The transformation works as follows:
      1. Find all square hollow frames.
         A frame is a square whose border is made of one single non-zero color.
         The interior must not contain that border color and must contain at
         least one 0.
      2. Keep the most common interior size among the valid frames.
         Ties are broken by preferring smaller interiors, then by stable order.
      3. For every selected frame, take its interior and normalize it:
         - 0 and the frame border color become 0.
         - Color 6 remains in a separate "6" mask.
         - Every other non-zero color is treated as a generic "other" mask.
      4. XOR all selected interiors together in the two separate masks.
      5. Build the output:
         - If the generic mask is active, output 2.
         - Else if the 6 mask is active, output 6.
         - Otherwise output 0.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    if H < 3 or W < 3:
        return np.zeros_like(grid)

    colors = sorted({int(v) for v in grid.flat if int(v) != 0})

    # Prefix sums for fast O(1) rectangle queries.
    zero_prefix = np.zeros((H + 1, W + 1), dtype=np.int32)
    zero_prefix[1:, 1:] = (grid == 0).astype(np.int32).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(prefix, r1, c1, r2, c2):
        """Sum over inclusive rectangle [r1, r2] x [c1, c2]."""
        return int(
            prefix[r2 + 1, c2 + 1]
            - prefix[r1, c2 + 1]
            - prefix[r2 + 1, c1]
            + prefix[r1, c1]
        )

    frames = []

    # Detect square frames of every color.
    # Brute force over top-left corner and side length is fine for ARC-sized grids.
    for color in colors:
        mask = (grid == color).astype(np.int32)
        prefix = np.zeros((H + 1, W + 1), dtype=np.int32)
        prefix[1:, 1:] = mask.cumsum(axis=0).cumsum(axis=1)

        max_side = min(H, W)

        for side in range(3, max_side + 1):
            inner = side - 2
            if inner <= 0:
                continue

            # Required number of cells on the border of a side x side square.
            border_area = 2 * side + 2 * (side - 2)

            for r1 in range(H - side + 1):
                r2 = r1 + side - 1

                # Quick top/bottom row check.
                if rect_sum(prefix, r1, 0, r1, W - 1) == 0:
                    continue
                if rect_sum(prefix, r2, 0, r2, W - 1) == 0:
                    continue

                for c1 in range(W - side + 1):
                    c2 = c1 + side - 1

                    # Top and bottom rows must be completely the frame color.
                    top_ok = rect_sum(prefix, r1, c1, r1, c2) == side
                    if not top_ok:
                        continue
                    bottom_ok = rect_sum(prefix, r2, c1, r2, c2) == side
                    if not bottom_ok:
                        continue

                    # Left and right columns must be completely the frame color.
                    left_ok = rect_sum(prefix, r1, c1, r2, c1) == side
                    if not left_ok:
                        continue
                    right_ok = rect_sum(prefix, r1, c2, r2, c2) == side
                    if not right_ok:
                        continue

                    # The entire border must be the frame color.
                    border_count = (
                        rect_sum(prefix, r1, c1, r1, c2)
                        + rect_sum(prefix, r2, c1, r2, c2)
                        + rect_sum(prefix, r1 + 1, c1, r2 - 1, c1)
                        + rect_sum(prefix, r1 + 1, c2, r2 - 1, c2)
                    )
                    if border_count != border_area:
                        continue

                    # Interior must not contain the border color.
                    if inner > 0:
                        interior_border_color = rect_sum(
                            prefix,
                            r1 + 1, c1 + 1,
                            r2 - 1, c2 - 1
                        )
                        if interior_border_color != 0:
                            continue

                        # Require at least one zero inside, matching the examples.
                        interior_zeros = rect_sum(
                            zero_prefix,
                            r1 + 1, c1 + 1,
                            r2 - 1, c2 - 1
                        )
                        if interior_zeros == 0:
                            continue

                    frames.append((color, r1, c1, side))

    if not frames:
        return np.zeros_like(grid)

    # Group frames by interior size.
    groups = {}
    for color, r1, c1, side in frames:
        inner = side - 2
        if inner <= 0:
            continue
        key = (inner, inner)
        groups.setdefault(key, []).append((color, r1, c1, side))

    if not groups:
        return np.zeros_like(grid)

    # Choose the most common interior size; tie-break by smaller size.
    best_key = min(
        groups.items(),
        key=lambda kv: (-len(kv[1]), kv[0][0] + kv[0][1])
    )[0]

    ih, iw = best_key
    selected = groups[best_key]

    mask_6 = np.zeros((ih, iw), dtype=bool)
    mask_other = np.zeros((ih, iw), dtype=bool)

    for color, r1, c1, side in selected:
        interior = grid[r1 + 1:r1 + 1 + ih, c1 + 1:c1 + 1 + iw]

        for i in range(ih):
            for j in range(iw):
                v = int(interior[i, j])

                # Ignore background zeros and the frame's border color.
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