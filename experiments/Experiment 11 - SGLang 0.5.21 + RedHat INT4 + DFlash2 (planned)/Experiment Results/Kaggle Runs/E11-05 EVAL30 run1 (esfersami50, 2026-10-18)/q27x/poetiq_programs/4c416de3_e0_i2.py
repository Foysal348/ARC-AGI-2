import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete zero-framed corner motifs.

    The input contains rectangles whose boundaries are mostly 0s.  Colored cells
    may appear on the four interior diagonals from the rectangle corners.  The
    output completes the motif for each such diagonal:

    - If a seed color is found at diagonal distance d from a corner:
      * paint the diagonal from the corner to the seed;
      * paint the outside corner block opposite the interior diagonal.
        For d == 1 this is the 3-cell L: corner + two adjacent outside cells.
        For d >= 2 this is the 2x2 block immediately outside the corner.

    Existing non-background, non-zero pixels are not overwritten.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Background is the most frequent color.
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    out = grid.copy()
    zero = (grid == 0)

    def can_place(r, c, color):
        """True if cell is within bounds and not an existing important pixel."""
        if not (0 <= r < H and 0 <= c < W):
            return False
        v = out[r, c]
        return v == 0 or v == bg

    def paint(r, c, color):
        if can_place(r, c, color):
            out[r, c] = color

    # Find maximal rectangular regions whose border is zero OR already part of
    # a motif.  We treat border cells as valid if they are zero, background, or
    # already colored in the output (motif completion may have overwritten them).
    # This avoids failing when a corner is already occupied by its motif color.
    frames = []

    def is_frame(r0, c0, r1, c1):
        if r1 - r0 < 2 or c1 - c0 < 2:
            return False
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            return False

        # Top/bottom rows: every cell must be zero or already painted as a
        # non-background motif color.
        for c in range(c0, c1 + 1):
            if not (zero[r0, c] or out[r0, c] != bg):
                return False
            if not (zero[r1, c] or out[r1, c] != bg):
                return False

        # Left/right columns.
        for r in range(r0, r1 + 1):
            if not (zero[r, c0] or out[r, c0] != bg):
                return False
            if not (zero[r, c1] or out[r, c1] != bg):
                return False

        return True

    # Collect candidate maximal frames starting from zero cells.
    for r0 in range(H):
        for c0 in range(W):
            if not zero[r0, c0]:
                continue

            # Extend down while left border can remain zero/motif.
            r1 = r0
            while r1 + 1 < H and (zero[r1 + 1, c0] or out[r1 + 1, c0] != bg):
                r1 += 1
            if r1 - r0 < 2:
                continue

            # Extend right while top border can remain zero/motif.
            c1 = c0
            while c1 + 1 < W and (zero[r0, c1 + 1] or out[r0, c1 + 1] != bg):
                c1 += 1
            if c1 - c0 < 2:
                continue

            if is_frame(r0, c0, r1, c1):
                frames.append((r0, c0, r1, c1))

    # Keep only maximal frames, removing contained duplicates.
    frames.sort(key=lambda f: -((f[2] - f[0]) * (f[3] - f[1])))
    unique_frames = []
    for f in frames:
        r0, c0, r1, c1 = f
        contained = False
        for g in unique_frames:
            gr0, gc0, gr1, gc1 = g
            if gr0 <= r0 and gc0 <= c0 and r1 <= gr1 and c1 <= gc1:
                contained = True
                break
        if not contained:
            unique_frames.append(f)

    def paint_corner(cr, cc, dr, dc, dist, color):
        """
        Paint one corner motif.

        (cr, cc) : corner of the zero frame.
        (dr, dc) : unit step from the corner into the interior diagonal.
        dist     : diagonal distance from corner to the seed pixel.
        color    : motif color.
        """
        # Diagonal path from corner to seed, inclusive.
        for k in range(dist + 1):
            paint(cr + dr * k, cc + dc * k, color)

        # Outside block in the opposite quadrant.
        odr, odc = -dr, -dc

        if dist == 1:
            # Three-cell L: corner plus the two orthogonal outside cells.
            paint(cr, cc, color)
            paint(cr + odr, cc, color)
            paint(cr, cc + odc, color)
        else:
            # 2x2 outside block.
            for i in (0, 1):
                for j in (0, 1):
                    paint(cr + odr * i, cc + odc * j, color)

    # For each frame and corner, walk the interior diagonal and complete the
    # first non-background, non-zero seed found.
    for r0, c0, r1, c1 in unique_frames:
        corners = [
            (r0, c0,  1,  1),   # top-left
            (r0, c1,  1, -1),   # top-right
            (r1, c0, -1,  1),   # bottom-left
            (r1, c1, -1, -1),   # bottom-right
        ]

        for cr, cc, dr, dc in corners:
            best_dist = None
            best_color = None

            max_steps = min(r1 - r0, c1 - c0)
            for k in range(1, max_steps + 1):
                rr = cr + dr * k
                cc_pos = cc + dc * k

                val = int(grid[rr, cc_pos])

                # Skip zero cells inside the diagonal; stop if we leave interior.
                if val == 0:
                    continue

                if val != bg:
                    best_dist = k
                    best_color = val
                    break

            if best_dist is not None and best_color is not None:
                paint_corner(cr, cc, dr, dc, best_dist, best_color)

    return out