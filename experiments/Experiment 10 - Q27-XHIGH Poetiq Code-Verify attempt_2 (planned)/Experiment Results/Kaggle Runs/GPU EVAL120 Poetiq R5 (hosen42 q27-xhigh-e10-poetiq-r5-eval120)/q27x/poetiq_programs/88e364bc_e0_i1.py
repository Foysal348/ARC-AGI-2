import numpy as np
from math import copysign


def _sign(x: float) -> int:
    """Return -1, 0, or 1 for the sign of x, ignoring tiny floating errors."""
    if x > 1e-9:
        return 1
    if x < -1e-9:
        return -1
    return 0


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    """Build a 2D prefix sum from a boolean/integer mask."""
    p = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    p[1:, 1:] = mask.astype(int)
    p = p.cumsum(axis=0).cumsum(axis=1)
    return p


def _rect_sum(p: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
    """Sum over inclusive rectangle [r1..r2] x [c1..c2] using prefix p."""
    return int(
        p[r2 + 1, c2 + 1]
        - p[r1, c2 + 1]
        - p[r2 + 1, c1]
        + p[r1, c1]
    )


def _find_indicator_directions(grid: np.ndarray):
    """
    Find the movement direction encoded by each solid color block containing
    1/2 (or 5/2 when the block color is 1).

    For a block color C:
      - if C != 1, direction is from centroid of 1s to centroid of 2s
      - if C == 1, direction is from centroid of 5s to centroid of 2s

    Returns a dict: color -> (dr, dc), where dr/dc are in {-1,0,1}.
    """
    H, W = grid.shape
    directions = {}

    # Candidate frame colors: non-background, not the movable 4, and not 2.
    for C in sorted(set(np.unique(grid).tolist())):
        if C in (0, 2, 4):
            continue

        from_color = 5 if C == 1 else 1
        to_color = 2

        # First try zero-free solid blocks. If none exists, retry allowing 0.
        found = False
        for allow_zero in (False, True):
            allowed = {C, from_color, to_color}
            if allow_zero:
                allowed.add(0)

            bad = np.zeros((H, W), dtype=bool)
            for r in range(H):
                for c in range(W):
                    if grid[r, c] not in allowed:
                        bad[r, c] = True

            p_bad = _prefix_sum(bad)
            p_from = _prefix_sum(grid == from_color)
            p_to = _prefix_sum(grid == to_color)

            best_area = 0
            best_rect = None

            for r1 in range(H):
                if grid[r1, 0] != C:  # cheap rejection only sometimes, but okay
                    pass
                for c1 in range(W):
                    if grid[r1, c1] != C:
                        continue
                    for r2 in range(r1 + 2, H):  # at least 3x3
                        if grid[r2, c1] != C:
                            continue
                        for c2 in range(c1 + 2, W):
                            if grid[r1, c2] != C or grid[r2, c2] != C:
                                continue

                            if _rect_sum(p_bad, r1, c1, r2, c2) != 0:
                                continue

                            cnt_from = _rect_sum(p_from, r1, c1, r2, c2)
                            cnt_to = _rect_sum(p_to, r1, c1, r2, c2)

                            if cnt_from == 0 or cnt_to == 0:
                                continue

                            area = (r2 - r1 + 1) * (c2 - c1 + 1)
                            if area > best_area:
                                best_area = area
                                best_rect = (r1, c1, r2, c2)

            if best_rect is not None:
                r1, c1, r2, c2 = best_rect

                frs, fcs, trs, tcs = [], [], [], []
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        if grid[r, c] == from_color:
                            frs.append(r)
                            fcs.append(c)
                        elif grid[r, c] == to_color:
                            trs.append(r)
                            tcs.append(c)

                if len(frs) and len(trs):
                    from_r = sum(frs) / len(frs)
                    from_c = sum(fcs) / len(fcs)
                    to_r = sum(trs) / len(trs)
                    to_c = sum(tcs) / len(tcs)

                    dr = _sign(to_r - from_r)
                    dc = _sign(to_c - from_c)

                    if dr != 0 or dc != 0:
                        directions[C] = (dr, dc)
                        found = True
                        break

        # If no block was found for this color, simply do not use it.

    return directions


def _nearest_direction_color(grid: np.ndarray, r: int, c: int, directions: dict):
    """
    Find the nearest color that has a defined direction, using expanding
    Chebyshev rings around (r, c). Colors without directions are ignored.
    """
    H, W = grid.shape
    max_r = max(H, W)

    for radius in range(1, max_r + 1):
        best = None  # (euclidean_sq, color)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                val = grid[rr, cc]
                if val == 0 or val == 4:
                    continue
                if val in directions:
                    key = (dr * dr + dc * dc, val)
                    if best is None or key < best:
                        best = key
        if best is not None:
            return best[1]

    # Fallback: minimum squared distance to any direction-colored pixel.
    best = None
    for C in directions:
        ys, xs = np.where(grid == C)
        if len(ys) == 0:
            continue
        d2 = np.min((ys - r) ** 2 + (xs - c) ** 2)
        key = (int(d2), int(C))
        if best is None or key < best:
            best = key

    return None if best is None else best[1]


def _find_gap_for_wall_shift(
    r: int, c: int, dr: int, dc: int, color: int, grid: np.ndarray
):
    """
    Special wall-shift case.

    If a 4 is moving orthogonally and would stop just before a wall pixel O
    of the associated color, and O is the end of a line of that color with a
    one-cell gap to another pixel of the same color, then O can slide into the
    gap. Returns the gap cell if such a slide is possible.

    Pattern checked for each perpendicular direction P:
        O + P   == color
        O + 2P  == 0
        O + 3P  == color
    """
    H, W = grid.shape

    # Only orthogonal movement has well-defined perpendicular wall lines here.
    if (dr == 0) == (dc == 0):
        return None

    if dr != 0:
        ps = [(0, 1), (0, -1)]
    else:
        ps = [(1, 0), (-1, 0)]

    for pr, pc in ps:
        r1, c1 = r + pr, c + pc
        r2, c2 = r + 2 * pr, c + 2 * pc
        r3, c3 = r + 3 * pr, c + 3 * pc

        if not (0 <= r1 < H and 0 <= c1 < W):
            continue
        if not (0 <= r2 < H and 0 <= c2 < W):
            continue
        if not (0 <= r3 < H and 0 <= c3 < W):
            continue

        if grid[r1, c1] == color and grid[r2, c2] == 0 and grid[r3, c3] == color:
            return (r2, c2)

    return None


def _move_one_4(
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    color: int,
    grid: np.ndarray
):
    """
    Move one 4 from its start position in direction (dr, dc).

    Returns:
      target: (r, c)
      wall_shift: (from_r, from_c, gap_r, gap_c) or None
    """
    H, W = grid.shape
    step = 0

    while True:
        nr = start_r + dr * (step + 1)
        nc = start_c + dc * (step + 1)

        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            target = (start_r + dr * step, start_c + dc * step)
            return target, None

        val = grid[nr, nc]

        # 0 is empty; 4 is treated as passable because all original 4s are
        # cleared before moving, and previous moved 4s should not block.
        if val == 0 or val == 4:
            step += 1
            continue

        # First real obstacle.
        target = (start_r + dr * step, start_c + dc * step)

        if val == color and (dr != 0) != (dc != 0):
            gap = _find_gap_for_wall_shift(nr, nc, dr, dc, color, grid)
            if gap is not None:
                return (nr, nc), (nr, nc, gap[0], gap[1])

        return target, None


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:

    1. Small solid blocks of a color C encode a direction.
       - For C != 1: direction is from the centroid of color 1 to the centroid
         of color 2 inside that block.
       - For C == 1: direction is from the centroid of color 5 to the centroid
         of color 2 inside that block.

    2. Each 4 is assigned the nearest encoded color (wall color).

    3. Each 4 moves in the assigned direction as far as possible through
       empty cells. It normally stops immediately before the first non-zero,
       non-4 obstacle.

    4. Special case: if a 4 moving orthogonally would stop just before a wall
       pixel of its assigned color, and that wall pixel is at the end of a
       line of the same color with a one-cell gap to another pixel of the same
       color, the wall pixel slides into that gap and the 4 occupies the wall
       pixel's original position.
    """
    if grid.size == 0:
        return grid.copy()

    out = np.array(grid, copy=True)
    H, W = out.shape

    directions = _find_indicator_directions(out)
    if not directions:
        return out

    # Collect original 4 positions and clear them so they do not block paths.
    fours = []
    for r in range(H):
        for c in range(W):
            if out[r, c] == 4:
                fours.append((r, c))
                out[r, c] = 0

    if not fours:
        return out

    # Process 4s in row-major order.
    for r, c in sorted(fours):
        color = _nearest_direction_color(out, r, c, directions)
        if color is None or color not in directions:
            # No usable direction: leave this 4 removed.
            continue

        dr, dc = directions[color]
        target, wall_shift = _move_one_4(r, c, dr, dc, color, out)

        if wall_shift is not None:
            wr, wc, gr, gc = wall_shift
            # Slide the wall pixel into the gap, then place the 4 in its place.
            out[gr, gc] = color
            out[wr, wc] = 0

        # If target is already occupied by another moved 4, keep the previous
        # one to preserve the number of 4s. This should not happen in normal
        # task inputs.
        if out[target[0], target[1]] != 4:
            out[target[0], target[1]] = 4

    return out