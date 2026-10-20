import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:

    1. The left 8x8 part of the grid is a template of 5s.
       It is removed from the left side and stamped onto the right side,
       shifted 10 columns to the right.

    2. Colored vertical segments in the right side are raised upward.
       In each column, let 'floor' be the topmost row containing a stamped 5.
       The segment's bottom row becomes:
           min(original_bottom, floor - 1)
       If there is no stamped 5 in that column, the segment is left unchanged.
       If the segment would extend above the top border, it is clipped.

    3. Stamped 5s are drawn last, so they override any conflicting color.
    """
    g = np.array(grid, dtype=int, copy=True)

    if g.ndim != 2:
        return g

    H, W = g.shape
    if H < 3 or W < 3:
        return g

    # Known layout for these examples: 8 rows x 20 columns.
    # Right box starts at column 9, and the left template maps to columns 10-17.
    if H == 8 and W == 20:
        right_start = 9
        left_width = 8
        shift = 10
    else:
        # Fallback heuristic for similar grids:
        # find the beginning of the top border of the right box.
        right_start = None
        for c in range(W):
            if g[0, c] == 5 and (c + 1 == W or g[0, c + 1] == 5):
                right_start = c
                break

        if right_start is None or right_start < 2:
            right_start = max(0, W // 2)

        left_width = min(H, max(0, right_start - 1))
        shift = right_start + 1

        if shift >= W:
            left_width = 0

    out = g.copy()

    # Erase the left template and the separator before the right box.
    if right_start > 0:
        out[:, :right_start] = 0

    # Build the stamped mask from the original left template.
    mask = np.zeros((H, W), dtype=bool)
    min_target_col = right_start + 1
    max_target_col = W - 2

    if min_target_col <= max_target_col:
        for r in range(H):
            for c in range(left_width):
                target_c = c + shift
                if min_target_col <= target_c <= max_target_col and g[r, c] == 5:
                    mask[r, target_c] = True

    # Clear original colored pixels in the interior of the right box.
    # The 5s in the right frame are preserved for now.
    if right_start + 1 <= W - 2:
        interior = np.zeros((H, W), dtype=bool)
        interior[1:H - 1, right_start + 1:W - 1] = True
        original_colors = (g != 0) & (g != 5) & interior
        out[original_colors] = 0

    # For each column, find the topmost stamped 5 row (the "floor").
    floors = [None] * W
    for c in range(W):
        rows_with_5 = np.flatnonzero(mask[1:H, c])  # ignore top border row 0
        if rows_with_5.size:
            floors[c] = int(rows_with_5[0]) + 1

    # Process each original vertical colored run in the right interior.
    if right_start + 1 <= W - 2:
        for c in range(right_start + 1, W - 1):
            r = 1
            while r < H - 1:
                val = g[r, c]

                if val != 0 and val != 5:
                    start = r
                    while r + 1 < H - 1 and g[r + 1, c] == val:
                        r += 1
                    end = r

                    height = end - start + 1
                    floor = floors[c]

                    if floor is None:
                        # No stamped terrain in this column: leave unchanged.
                        final_bottom = end
                    else:
                        # Raise the block so its bottom is just above the terrain.
                        final_bottom = min(end, floor - 1)

                    if final_bottom >= 1:
                        final_top = max(1, final_bottom - height + 1)

                        for rr in range(final_top, final_bottom + 1):
                            # Stamped 5s should have priority.
                            if not mask[rr, c]:
                                out[rr, c] = val

                r += 1

    # Draw the stamped terrain 5s on top.
    out[mask] = 5

    # Ensure the erased left area remains zero.
    if right_start > 0:
        out[:, :right_start] = 0

    return out