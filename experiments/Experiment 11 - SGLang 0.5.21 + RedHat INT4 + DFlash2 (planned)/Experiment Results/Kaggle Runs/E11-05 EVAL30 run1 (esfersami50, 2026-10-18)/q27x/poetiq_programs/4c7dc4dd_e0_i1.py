import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by:
      1. Detecting hollow rectangular frames.
         A frame is a rectangle whose border is one single colour, whose interior
         does not contain that border colour, and whose interior contains at least
         one 0.
      2. Choosing the most common interior size among detected frames.
      3. XOR-ing all frames with that interior size.
         - Cells coloured 6 are tracked separately.
         - All other non-zero, non-frame colours are tracked as a single generic mask.
         - If the generic mask is active, output 2.
         - If the 6 mask is active and the generic mask is inactive, output 6.
         - Otherwise output 0.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    colors = list(set(int(x) for x in grid.flat if int(x) != 0))

    zero_mask = (grid == 0)
    zero_prefix = np.zeros((H + 1, W + 1), dtype=np.int32)
    zero_prefix[1:, 1:] = zero_mask.astype(np.int32).cumsum(axis=0).cumsum(axis=1)

    def prefix_sum(mask, r1, r2, c1, c2):
        """Sum of mask over rows [r1, r2], cols [c1, c2]."""
        return (
            mask[r2 + 1, c2 + 1]
            - mask[r1, c2 + 1]
            - mask[r2 + 1, c1]
            + mask[r1, c1]
        )

    candidates = []

    for color in colors:
        if color == 0:
            continue

        mask = (grid == color).astype(np.int32)
        prefix = np.zeros((H + 1, W + 1), dtype=np.int32)
        prefix[1:, 1:] = mask.cumsum(axis=0).cumsum(axis=1)

        # Precompute row/col prefixes for faster border checks.
        row_prefix = np.zeros((H + 1, W + 1), dtype=np.int32)
        row_prefix[1:, 1:] = mask.cumsum(axis=1)

        col_prefix = np.zeros((H + 1, W + 1), dtype=np.int32)
        col_prefix[1:, 1:] = mask.cumsum(axis=0)

        for r1 in range(H - 1):
            # Top border must be a contiguous run of `color`.
            row = (grid[r1] == color)
            c1 = 0
            while c1 < W:
                if not row[c1]:
                    c1 += 1
                    continue

                c2 = c1
                while c2 + 1 < W and row[c2 + 1]:
                    c2 += 1

                if c2 - c1 >= 1:
                    # Try bottom rows.
                    for r2 in range(r1 + 1, H):
                        height = r2 - r1
                        width = c2 - c1

                        # Bottom row must match exactly.
                        if row_prefix[r2 + 1, c2 + 1] - row_prefix[r2 + 1, c1] != width:
                            # If r2 is too small, continue; if too large, stop?
                            # For simplicity, keep scanning.
                            pass
                        else:
                            # Left and right borders must be continuous.
                            left_ok = (
                                col_prefix[r2 + 1, c1 + 1] - col_prefix[r1, c1 + 1] == height
                            )
                            right_ok = (
                                col_prefix[r2 + 1, c2 + 1] - col_prefix[r1, c2 + 1] == height
                            )

                            if left_ok and right_ok:
                                # Interior must not contain the frame colour.
                                interior_has_frame_color = (
                                    prefix_sum(mask, r1 + 1, r2 - 1, c1 + 1, c2 - 1)
                                )
                                if interior_has_frame_color == 0:
                                    # Require at least one 0 in the interior.
                                    if (
                                        r2 > r1 + 1 and c2 > c1 + 1 and
                                        prefix_sum(zero_prefix, r1 + 1, r2 - 1, c1 + 1, c2 - 1) > 0
                                    ):
                                        candidates.append((color, r1, r2, c1, c2))

                c1 = c2 + 1

    if not candidates:
        return np.zeros_like(grid)

    # Group by interior size.
    groups = {}
    for color, r1, r2, c1, c2 in candidates:
        ih, iw = r2 - r1 - 1, c2 - c1 - 1
        if ih <= 0 or iw <= 0:
            continue
        key = (ih, iw)
        groups.setdefault(key, []).append((color, r1, r2, c1, c2))

    # Choose the group with the most frames.
    best_key = max(
        groups.items(),
        key=lambda kv: (len(kv[1]), kv[0][0] * kv[0][1])
    )[0]
    frames = groups[best_key]
    ih, iw = best_key

    mask_6 = np.zeros((ih, iw), dtype=bool)
    mask_other = np.zeros((ih, iw), dtype=bool)

    for color, r1, r2, c1, c2 in frames:
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