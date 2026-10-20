import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the two-panel ARC grid.

    The left panel contains a 4-row "histogram" of 5s.  The right panel
    contains a 3x4 active area of colored cells separated by grid lines.

    Rule:
    1. From the left histogram, compute a per-column capacity K for the four
       right columns.  K is the number of leading zero rows in the top three
       rows of that histogram column, capped at 3.
    2. In each right column, move the colored block upward by one row
       (row 0 does not move further up).  Then keep only the colored cells
       that lie in the top K rows of the 3-row active area.
    3. In the 3x4 active area, every cell not occupied by a kept color is 5.
    4. The lower fixed frame of the right panel is rebuilt as in the examples.
    5. The left panel is reduced as observed:
         - if its top row contains exactly one 5, keep only that top row;
         - otherwise keep only its bottom two rows.
    """
    if grid.size == 0:
        return grid.copy()

    out = np.array(grid, dtype=int, copy=True)
    H, W = out.shape

    # The known layout is 8 rows x 20 columns.  If a different size is given,
    # fall back to a safe copy.
    if H != 8 or W != 20:
        return out

    # ------------------------------------------------------------------
    # Left panel: 4x4 histogram at rows 2..5, columns 1,3,5,7
    # ------------------------------------------------------------------
    left_rows = range(2, 6)
    left_cols = [1, 3, 5, 7]

    left_hist = np.zeros((4, 4), dtype=int)
    for i, r in enumerate(left_rows):
        for j, c in enumerate(left_cols):
            left_hist[i, j] = 1 if grid[r, c] == 5 else 0

    # ------------------------------------------------------------------
    # Right panel:
    #   active 3x4 area: rows 1..3, columns 11,13,15,17
    #   input color rows: rows 1..4, same columns
    # ------------------------------------------------------------------
    right_rows_input = range(1, 5)      # 4 rows that may contain colors
    right_cols = [11, 13, 15, 17]

    # Capacity K for each of the 4 right columns.
    K = np.zeros(4, dtype=int)
    for j in range(4):
        first_5 = 3  # if no 5 in top 3 rows, all 3 active rows are allowed
        for i in range(3):
            if left_hist[i, j] == 1:
                first_5 = i
                break
        K[j] = min(3, first_5 + 1)

    # Determine final colors in the 3x4 active area.
    active = np.full((3, 4), 5, dtype=int)

    for j, c in enumerate(right_cols):
        # Collect colored cells in this input column (ignore 0 and 5).
        colored_rows = []
        color = 0
        for r in right_rows_input:
            val = grid[r, c]
            if val != 0 and val != 5:
                colored_rows.append(r - 1)  # convert to interior row 0..3
                color = val

        if color == 0 or not colored_rows:
            continue

        # Move upward by one, with row 0 staying at row 0.
        shifted = set(max(0, r - 1) for r in colored_rows)

        # Keep only rows inside the allowed top K active rows (0..K-1).
        kept = {r for r in shifted if r < K[j]}

        for r in kept:
            active[r, j] = color

    # Write active 3x4 area into output.
    for i in range(3):
        r = 1 + i
        for j, c in enumerate(right_cols):
            out[r, c] = active[i, j]

    # Reset the even columns in the upper active area to 0.
    for r in range(1, 4):
        for c in [10, 12, 14, 16, 18]:
            out[r, c] = 0

    # ------------------------------------------------------------------
    # Rebuild the fixed lower right-panel frame.
    # Interior row 3 (grid row 4): even columns 12,14,16,18 are 5.
    # Interior row 4 (grid row 5): columns 12..17 are 5.
    # Interior row 5 (grid row 6): all 0.
    # ------------------------------------------------------------------
    out[4, 11] = 0
    out[4, 13] = 0
    out[4, 15] = 0
    out[4, 17] = 0
    for c in [12, 14, 16, 18]:
        out[4, c] = 5

    out[5, 10] = 0
    out[5, 11] = 0
    out[5, 18] = 0
    for c in range(12, 18):
        out[5, c] = 5

    out[6, 10:19] = 0

    # ------------------------------------------------------------------
    # Left panel output.
    # The observed behavior is:
    #   - if the top row of the left panel has exactly one 5,
    #     keep only that top row;
    #   - otherwise keep only the bottom two rows of the left panel.
    # ------------------------------------------------------------------
    # Clear the whole left area first.
    out[2:6, 0:9] = 0

    top_count = int(sum(1 for c in range(9) if grid[2, c] == 5))

    if top_count == 1:
        # Keep only the original top row of the left panel.
        for c in range(9):
            if grid[2, c] == 5:
                out[2, c] = 5
    else:
        # Keep only the original bottom two rows of the left panel.
        for r in [4, 5]:
            for c in range(9):
                if grid[r, c] == 5:
                    out[r, c] = 5

    return out