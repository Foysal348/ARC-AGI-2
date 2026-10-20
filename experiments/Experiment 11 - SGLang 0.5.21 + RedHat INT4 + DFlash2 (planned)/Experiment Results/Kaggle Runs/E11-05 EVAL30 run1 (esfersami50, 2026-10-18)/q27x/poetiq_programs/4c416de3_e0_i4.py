import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete corner motifs around 0-regions.
    
    For each 0-corner (a 0-cell with zero arms in two perpendicular directions),
    look along the interior diagonal for a colored seed (non-background, non-zero)
    at distance 1 or 2. If found, paint a motif:
      - d=1: thick style (6 cells) if a 5x5 zero frame with corner seeds exists,
             otherwise thin style (4 cells: corner + 2 outside + seed).
      - d=2: 2x2 block outside the corner, plus the intermediate diagonal cell
             only if the diagonal direction is downward (positive row delta).
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    if H < 2 or W < 2:
        return grid.copy()

    out = grid.copy()

    # Background = most frequent color
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    zero = (grid == 0)

    def can_paint(r, c):
        return 0 <= r < H and 0 <= c < W and out[r, c] in (bg, 0)

    def paint(r, c, color):
        if can_paint(r, c):
            out[r, c] = color

    def zero_run(r, c, dr, dc):
        """Count consecutive zero cells starting at (r,c) in direction (dr,dc)."""
        length = 0
        rr, cc = r, c
        while 0 <= rr < H and 0 <= cc < W and zero[rr, cc]:
            length += 1
            rr += dr
            cc += dc
        return length

    # --- Find all 0-corners ---
    corners = []
    seen = set()
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for r in range(H):
        for c in range(W):
            if not zero[r, c]:
                continue
            for i in range(4):
                for j in range(i + 1, 4):
                    dr1, dc1 = dirs[i]
                    dr2, dc2 = dirs[j]
                    # Must be perpendicular arms
                    if dr1 * dr2 + dc1 * dc2 != 0:
                        continue
                    if zero_run(r, c, dr1, dc1) >= 2 and zero_run(r, c, dr2, dc2) >= 2:
                        dr_in = dr1 + dr2
                        dc_in = dc1 + dc2
                        if dr_in != 0 and dc_in != 0:
                            key = (r, c, dr_in, dc_in)
                            if key not in seen:
                                seen.add(key)
                                corners.append((r, c, dr_in, dc_in))

    # --- Determine thick vs thin d=1 style ---
    def has_thick_frame():
        """Check for a 5x5 all-zero-border frame with >=2 corner seeds at d=1."""
        if H < 5 or W < 5:
            return False
        for r0 in range(H - 4):
            for c0 in range(W - 4):
                r1, c1 = r0 + 4, c0 + 4
                if (np.all(zero[r0, c0:c1 + 1]) and
                    np.all(zero[r1, c0:c1 + 1]) and
                    np.all(zero[r0:r1 + 1, c0]) and
                    np.all(zero[r0:r1 + 1, c1])):
                    seeds = 0
                    for rr, cc, dr, dc in [(r0, c0, 1, 1), (r0, c1, 1, -1),
                                            (r1, c0, -1, 1), (r1, c1, -1, -1)]:
                        sr, sc = rr + dr, cc + dc
                        if 0 <= sr < H and 0 <= sc < W and grid[sr, sc] not in (0, bg):
                            seeds += 1
                    if seeds >= 2:
                        return True
        return False

    thick = has_thick_frame()

    # --- Process each corner ---
    for cr, cc, dr, dc in corners:
        od_r, od_c = -dr, -dc
        seed_d, seed_c = None, None

        # Search for seed at distance 1 or 2 along interior diagonal
        for k in range(1, 3):
            rr, cc2 = cr + dr * k, cc + dc * k
            if not (0 <= rr < H and 0 <= cc2 < W):
                break
            v = int(grid[rr, cc2])
            if v == 0:
                continue  # skip zeros, keep looking
            if v != bg:
                seed_d, seed_c = k, v
                break

        if seed_d is None:
            continue

        if seed_d == 1:
            if thick:
                # Thick: 6 cells = two 2x2 quadrants minus the corner
                paint(cr + od_r, cc + od_c, seed_c)
                paint(cr + od_r, cc, seed_c)
                paint(cr, cc + od_c, seed_c)
                paint(cr, cc + dc, seed_c)
                paint(cr + dr, cc, seed_c)
                paint(cr + dr, cc + dc, seed_c)
            else:
                # Thin: corner + 2 outside orthogonal + seed
                paint(cr, cc, seed_c)
                paint(cr + od_r, cc, seed_c)
                paint(cr, cc + od_c, seed_c)
                paint(cr + dr, cc + dc, seed_c)
        else:
            # d == 2: 2x2 block outside (including corner)
            paint(cr, cc, seed_c)
            paint(cr + od_r, cc + od_c, seed_c)
            paint(cr + od_r, cc, seed_c)
            paint(cr, cc + od_c, seed_c)
            # Intermediate diagonal cell: paint only when going downward
            if dr > 0:
                paint(cr + dr, cc + dc, seed_c)

    return out