import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete zero-corner corner motifs.

    The examples contain 0-cells that act as corners of zero lines/rectangles.
    A non-background seed color is placed on a diagonal from such a corner.
    The output completes the associated corner motif.

    There are three observed motif styles:

    1. Thick d=1 motif (Example 1 style):
       For a seed at diagonal distance 1, paint the six cells in the two
       opposite 2x2 quadrants around the 0-corner, leaving the 0-corner itself
       as 0.

    2. Thin d=1 motif (Example 2 style):
       For a seed at diagonal distance 1, paint the corner itself, the two
       outside orthogonal neighbours, and the seed.

    3. d>=2 motif (Example 3 style):
       Paint the diagonal path from the corner to the seed, and paint the
       2x2 block on the outside of the corner (including the corner).

    The code identifies all 0-corners, finds diagonal seeds, and completes
    motifs with the appropriate style.  Existing non-background pixels are not
    overwritten.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    out = grid.copy()

    # Background is the most frequent color.
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    zero = (grid == 0)

    def is_bg_or_zero(r, c):
        return 0 <= r < H and 0 <= c < W and (out[r, c] == bg or out[r, c] == 0)

    def can_paint(r, c, color):
        if not (0 <= r < H and 0 <= c < W):
            return False
        return out[r, c] in (bg, 0)

    def paint(r, c, color):
        if can_paint(r, c, color):
            out[r, c] = color

    def zero_run_length(r, c, dr, dc):
        """Length of consecutive zero cells starting at (r,c) in direction (dr,dc)."""
        length = 0
        rr, cc = r, c
        while 0 <= rr < H and 0 <= cc < W and zero[rr, cc]:
            length += 1
            rr += dr
            cc += dc
        return length

    # Find all zero cells that can be corners: they have zero runs in two
    # perpendicular directions.
    corners = []
    for r in range(H):
        for c in range(W):
            if not zero[r, c]:
                continue

            # Try four corner orientations.
            # (dr1,dc1), (dr2,dc2) are the directions of the zero arms.
            orientations = [
                ((0, 1), (1, 0)),    # top-left corner of a frame: right and down
                ((0, -1), (1, 0)),   # top-right corner: left and down
                ((0, 1), (-1, 0)),   # bottom-left corner: right and up
                ((0, -1), (-1, 0)),  # bottom-right corner: left and up
            ]

            for dr1, dc1 in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                for dr2, dc2 in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    if (dr1, dc1) == (dr2, dc2) or (dr1, dc1) == (-dr2, -dc2):
                        continue
                    # Require perpendicular arms.
                    if dr1 == dr2 and dc1 == dc2:
                        continue
                    if dr1 == 0 and dr2 == 0:
                        continue
                    if dc1 == 0 and dc2 == 0:
                        continue

                    len1 = zero_run_length(r, c, dr1, dc1)
                    len2 = zero_run_length(r, c, dr2, dc2)

                    # The corner itself is counted once; require at least one
                    # step in each arm, i.e. total run >= 2 in both directions.
                    if len1 >= 2 and len2 >= 2:
                        # Interior direction is the diagonal between the two arms.
                        dr_in = dr1 + dr2
                        dc_in = dc1 + dc2
                        if dr_in == 0 or dc_in == 0:
                            continue
                        corners.append((r, c, dr_in, dc_in, -dr_in, -dc_in))

    # Deduplicate corners.  Keep the most informative representation.
    seen = set()
    unique_corners = []
    for item in corners:
        key = (item[0], item[1], item[2], item[3])
        if key not in seen:
            seen.add(key)
            unique_corners.append(item)

    def find_seed(r, c, dr, dc):
        """
        Walk diagonally inside from a corner.  Return (dist, color) for the
        first non-background, non-zero cell found.
        """
        for k in range(1, min(H, W)):
            rr = r + dr * k
            cc = c + dc * k
            if not (0 <= rr < H and 0 <= cc < W):
                break

            val = int(grid[rr, cc])

            if val == 0:
                # Stop if we hit a zero that is part of the corner's frame.
                # This prevents running along another diagonal through zeros.
                if (rr == r or cc == c):
                    break
                continue

            if val != bg:
                return k, val

        return None, None

    def paint_thick_d1(corner_r, corner_c, dr_in, dc_in, od_r, od_c, color):
        """Example 1 style: two opposite 2x2 quadrants, corner remains 0."""
        # Outside 2x2 (opposite the interior direction), excluding corner.
        paint(corner_r + od_r, corner_c + od_c, color)
        paint(corner_r + od_r, corner_c, color)
        paint(corner_r, corner_c + od_c, color)

        # Inside 2x2 (interior direction), excluding corner.
        paint(corner_r, corner_c + dc_in, color)
        paint(corner_r + dr_in, corner_c, color)
        paint(corner_r + dr_in, corner_c + dc_in, color)

    def paint_thin_d1(corner_r, corner_c, dr_in, dc_in, od_r, od_c, color):
        """Example 2 style: corner + two outside orthogonal neighbours + seed."""
        paint(corner_r, corner_c, color)
        paint(corner_r + od_r, corner_c, color)
        paint(corner_r, corner_c + od_c, color)
        # Seed itself is already present, but paint it for completeness.
        paint(corner_r + dr_in, corner_c + dc_in, color)

    def paint_d_ge_2(corner_r, corner_c, dr_in, dc_in, od_r, od_c, dist, color):
        """
        Example 3 style: outside 2x2 block including the corner, plus diagonal
        path from the corner to the seed.
        """
        # Outside 2x2 including corner.
        paint(corner_r, corner_c, color)
        paint(corner_r + od_r, corner_c + od_c, color)
        paint(corner_r + od_r, corner_c, color)
        paint(corner_r, corner_c + od_c, color)

        # Diagonal path.
        for k in range(1, dist + 1):
            paint(corner_r + dr_in * k, corner_c + dc_in * k, color)

    # Determine which d=1 style to use.
    # The examples show that the style is correlated with the overall layout.
    # A robust heuristic: if the grid contains at least one full rectangular
    # zero frame whose interior is 3x3 (frame 5x5) and with four diagonal
    # seeds at its corners, use the thick d=1 style.  Otherwise use thin d=1.
    def has_small_full_frame_with_corner_seeds():
        # Look for a 5x5 full zero frame.
        for r0 in range(H - 4):
            for c0 in range(W - 4):
                r1, c1 = r0 + 4, c0 + 4
                if not (np.all(zero[r0, c0:c1 + 1]) and
                        np.all(zero[r1, c0:c1 + 1]) and
                        np.all(zero[r0:r1 + 1, c0]) and
                        np.all(zero[r0:r1 + 1, c1])):
                    continue

                # Check for diagonal seeds near each corner.
                seeds = []
                for rr, cc, dr, dc in [
                    (r0, c0, 1, 1),
                    (r0, c1, 1, -1),
                    (r1, c0, -1, 1),
                    (r1, c1, -1, -1),
                ]:
                    sr = rr + dr
                    sc = cc + dc
                    if 0 <= sr < H and 0 <= sc < W and grid[sr, sc] not in (0, bg):
                        seeds.append(int(grid[sr, sc]))

                if len(seeds) >= 2:
                    return True
        return False

    use_thick_d1 = has_small_full_frame_with_corner_seeds()

    for cr, cc, dr_in, dc_in, od_r, od_c in unique_corners:
        dist, color = find_seed(cr, cc, dr_in, dc_in)
        if dist is None or color is None:
            continue

        if dist == 1:
            if use_thick_d1:
                paint_thick_d1(cr, cc, dr_in, dc_in, od_r, od_c, color)
            else:
                paint_thin_d1(cr, cc, dr_in, dc_in, od_r, od_c, color)
        else:
            paint_d_ge_2(cr, cc, dr_in, dc_in, od_r, od_c, dist, color)

    return out