import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete the corner decorations associated with rectangular 0-borders.

    The task contains rectangular regions whose boundary is made of 0s.  In the
    input, some colored pixels appear inside those rectangles, usually on one of
    the two diagonal directions leading from a corner.  The output completes
    the corresponding corner decoration:

    1. Detect all rectangular 0-border frames.
    2. For each frame, examine the four diagonal rays inside the frame.
    3. Assign colored pixels to the closest corner along the diagonal ray.
    4. Paint the diagonal path from the corner to that pixel, and paint the
       corresponding outside corner block.

    The outside corner block is:
    - a 2x2 block when the diagonal segment has length at least 2;
    - a 3-cell L shape when the diagonal segment has length 1.
    """

    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Background is the most common color.
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    out = grid.copy()

    def is_zero_frame(r0, c0, r1, c1):
        """Check whether all border cells of the rectangle are 0."""
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            return False
        if r1 - r0 < 2 or c1 - c0 < 2:
            return False

        # top and bottom rows
        if np.any(grid[r0, c0:c1 + 1] != 0):
            return False
        if np.any(grid[r1, c0:c1 + 1] != 0):
            return False
        # left and right columns
        if np.any(grid[r0:r1 + 1, c0] != 0):
            return False
        if np.any(grid[r0:r1 + 1, c1] != 0):
            return False

        return True

    # Collect all maximal rectangular zero frames.
    frames = []
    zero = (grid == 0)

    for r0 in range(H):
        for c0 in range(W):
            if not zero[r0, c0]:
                continue
            # Extend as far right and down as possible while keeping a zero border.
            r1 = r0
            while r1 + 1 < H and zero[r1 + 1, c0]:
                r1 += 1
            if r1 - r0 < 2:
                continue

            c1 = c0
            while c1 + 1 < W and zero[r0, c1 + 1]:
                c1 += 1
            if c1 - c0 < 2:
                continue

            # We have candidate top-left, top-right, and bottom-left zero corners.
            # Verify the full rectangle; if valid, record it.
            if is_zero_frame(r0, c0, r1, c1):
                frames.append((r0, c0, r1, c1))

    # Remove duplicate / contained frames.  Keep maximal frames only.
    frames = sorted(frames, key=lambda f: -((f[2] - f[0]) * (f[3] - f[1])))
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

    # Helper: paint a corner motif.
    def paint_corner(r, c, dr, dc, length, color):
        """
        Paint the motif for one corner.

        (r, c)       : corner coordinate of the 0-frame.
        (dr, dc)     : unit step from corner into the interior diagonal.
        length       : number of diagonal cells from corner to the colored pixel.
        color        : color to paint.
        """
        # Diagonal path from corner to the pixel, inclusive.
        for k in range(length + 1):
            rr = r + dr * k
            cc = c + dc * k
            if 0 <= rr < H and 0 <= cc < W:
                # Do not overwrite existing non-background, non-zero colors.
                if out[rr, cc] in (0, bg):
                    out[rr, cc] = color

        # Outside corner block.
        # It lies in the quadrant opposite the interior direction.
        orr = -dr
        occ = -dc

        if length >= 2:
            # Full 2x2 outside block.
            for i in (0, 1):
                for j in (0, 1):
                    rr = r + orr * i
                    cc = c + occ * j
                    if 0 <= rr < H and 0 <= cc < W and out[rr, cc] in (0, bg):
                        out[rr, cc] = color
        else:
            # 3-cell L: corner plus the two outside orthogonal neighbors.
            for rr, cc in ((r, c), (r + orr, c), (r, c + occ)):
                if 0 <= rr < H and 0 <= cc < W and out[rr, cc] in (0, bg):
                    out[rr, cc] = color

    # For each frame, assign pixels on the four interior diagonals to corners.
    for r0, c0, r1, c1 in unique_frames:
        # Corner list with their interior diagonal direction.
        corners = [
            (r0, c0,  1,  1),   # top-left, interior down-right
            (r0, c1,  1, -1),   # top-right, interior down-left
            (r1, c0, -1,  1),   # bottom-left, interior up-right
            (r1, c1, -1, -1),   # bottom-right, interior up-left
        ]

        for cr, cc, dr, dc in corners:
            best_len = None
            best_color = None

            # Walk along the diagonal inside the frame.
            # Stop when hitting another non-background, non-zero color or border.
            for k in range(1, min(r1 - r0 + 1, c1 - c0 + 1)):
                rr = cr + dr * k
                cc = cc + dc * k

                if rr < r0 or rr > r1 or cc < c0 or cc > c1:
                    break

                val = int(grid[rr, cc])

                if val == 0:
                    # Border/gap on the diagonal; stop at the frame boundary.
                    if rr in (r0, r1) or cc in (c0, c1):
                        break
                    continue

                if val != bg:
                    if best_len is None:
                        best_len = k
                        best_color = val
                    break

            if best_len is not None and best_color is not None:
                paint_corner(cr, cc, dr, dc, best_len, best_color)

    return out