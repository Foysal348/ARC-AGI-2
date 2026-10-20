import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a small empty-centered cross into the corresponding recursive
    pinwheel/fractal line pattern.

    The input grid contains exactly one empty-centered cross made of 8s.
    The center is the missing cell where the horizontal and vertical arms
    meet.  The arm length is the number of 8-cells from that center to an
    arm tip.

    The output is built by drawing four recursive axis-aligned spirals,
    one for each arm of the input cross.  For each arm:
      * start at the cross center,
      * move outward along the arm,
      * repeatedly turn clockwise 90 degrees,
      * draw line segments whose lengths double after the first turn.

    The first outward segment is the arm itself.  If the arm length is 1,
    the first segment is extended by one cell before the first turn.
    Segments are clipped to the input grid.  When a segment is clipped,
    recursion does not continue from the clipped endpoint.
    """
    if grid is None:
        return np.zeros((0, 0), dtype=int)

    grid = np.asarray(grid)
    out = np.zeros_like(grid, dtype=grid.dtype)

    if grid.size == 0:
        return out

    h, w = grid.shape

    mask = (grid == 8)
    if not mask.any():
        return out

    # Find the horizontal arm row: it is the row containing the most 8s.
    row_counts = mask.sum(axis=1)
    center_row = int(np.argmax(row_counts))

    # In that row, the center is the zero gap between the left and right arms.
    cols = np.where(mask[center_row])[0]
    if cols.size < 2:
        # Degenerate fallback; use all 8 coordinates.
        ys, xs = np.where(mask)
        center_row = int(ys.mean())
        center_col = int(xs.mean())
    else:
        left = int(cols[0])
        right = int(cols[-1])
        center_col = (left + right) // 2

    # Determine arm length from the vertical arm.
    arm = 0
    r = center_row - 1
    while r >= 0 and mask[r, center_col]:
        arm += 1
        r -= 1

    if arm == 0:
        # If the vertical arm is not visible, fall back to horizontal half-width.
        if cols.size >= 2:
            arm = max(1, (int(cols[-1]) - int(cols[0]) + 1) // 2 - 1)
        else:
            return out

    # If arm length is 1, the first visible spiral segment is length 2.
    # For larger arms, the first segment is the input arm itself.
    first_len = arm if arm >= 2 else 2

    def draw_segment(r: int, c: int, dr: int, dc: int, length: int) -> bool:
        """
        Draw an axis-aligned segment of exactly `length` cells, starting from
        (r, c) but excluding the start cell.  Returns True iff the full segment
        fits inside the grid.
        """
        end_r = r + dr * length
        end_c = c + dc * length

        if not (0 <= end_r < h and 0 <= end_c < w):
            return False

        step_r = 1 if dr > 0 else -1 if dr < 0 else 0
        step_c = 1 if dc > 0 else -1 if dc < 0 else 0

        rr, cc = r, c
        for _ in range(length):
            rr += step_r
            cc += step_c
            out[rr, cc] = 8

        return True

    def spiral(r0: int, c0: int, dr: int, dc: int, length: int) -> None:
        """
        Draw a clockwise spiral.

        The first call uses `length` for the initial outward segment.
        After each successfully drawn segment, the direction turns clockwise
        and the length doubles.
        """
        # The first segment starts at the cross center and is drawn outward.
        if not draw_segment(r0, c0, dr, dc, length):
            return

        r, c = r0 + dr * length, c0 + dc * length
        length *= 2

        while True:
            # Turn clockwise: (dr, dc) -> (dc, -dr)
            dr, dc = dc, -dr
            if not draw_segment(r, c, dr, dc, length):
                return
            r += dr * length
            c += dc * length
            length *= 2

    # Four initial arms: up, down, left, right.
    # The order does not matter because we only set pixels to 8.
    spiral(center_row, center_col, -1, 0, first_len)   # up
    spiral(center_row, center_col,  1, 0, first_len)   # down
    spiral(center_row, center_col,  0,-1, first_len)   # left
    spiral(center_row, center_col,  0, 1, first_len)   # right

    return out