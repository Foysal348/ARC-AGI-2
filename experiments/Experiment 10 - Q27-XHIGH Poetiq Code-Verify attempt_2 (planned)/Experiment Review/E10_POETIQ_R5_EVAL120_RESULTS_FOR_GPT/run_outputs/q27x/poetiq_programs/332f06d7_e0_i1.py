import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move the 0-colored square to the matching target square.

    The input contains:
      - background color 3
      - a main object of color 1
      - one square of color 0
      - one square of color 2

    The 0 square has the same size as the 2 square.  In the output:
      - the original 0 square is filled with 1
      - a new 0 square is placed at the target position
      - if the target position is the existing 2 square, that 2 square becomes 0
      - otherwise, the 2 square remains unchanged

    Target selection:
      1. If there is a 1-colored square of the same size at the midpoint
         between the 0 square and the 2 square, use it.
      2. Otherwise, if the 2 square itself is at that midpoint, use it.
      3. Otherwise, search among square positions of the same size that are
         "central" relative to the 0 and 2 squares and prefer positions that
         contain the largest amount of color 1.
      4. Fallback to the 2 square if nothing better is found.
    """
    grid = np.array(grid, copy=True)
    h, w = grid.shape

    # Find all 0 and 2 pixels.
    zero_mask = grid == 0
    two_mask = grid == 2

    zero_cells = np.argwhere(zero_mask)
    two_cells = np.argwhere(two_mask)

    # Graceful fallback: if the expected markers are absent, return unchanged.
    if len(zero_cells) == 0 or len(two_cells) == 0:
        return grid

    # Bounding boxes for the 0 square and 2 square.
    r0_min, c0_min = zero_cells.min(axis=0)
    r0_max, c0_max = zero_cells.max(axis=0)
    r2_min, c2_min = two_cells.min(axis=0)
    r2_max, c2_max = two_cells.max(axis=0)

    # The two markers are assumed to be squares of the same size.
    sz = max(r0_max - r0_min + 1, c0_max - c0_min + 1,
             r2_max - r2_min + 1, c2_max - c2_min + 1)

    def is_square_block(mask, rmin, cmin, size):
        """Check whether mask contains exactly an axis-aligned size x size block."""
        if rmin + size > h or cmin + size > w:
            return False
        block = mask[rmin:rmin + size, cmin:cmin + size]
        if not np.all(block):
            return False
        # Also ensure no extra pixels of this color elsewhere.
        # For this task, there is only one block of each marker color.
        return True

    # Helper: count how many 1-cells a candidate square contains.
    def one_count(r, c):
        if r + sz > h or c + sz > w:
            return -1
        return int(np.sum(grid[r:r + sz, c:c + sz] == 1))

    # Candidate score: prefer positions near the midpoint between the 0 and 2 squares.
    # We use top-left coordinates for stable integer comparison.
    mid_r = (r0_min + r2_min) / 2.0
    mid_c = (c0_min + c2_min) / 2.0

    def midpoint_score(r, c):
        # Distance from the candidate square center to the midpoint.
        cr = r + (sz - 1) / 2.0
        cc = c + (sz - 1) / 2.0
        return (cr - mid_r) ** 2 + (cc - mid_c) ** 2

    # First, check exact midpoint positions (rounded).
    best = None
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r = int(round(mid_r)) + dr
            c = int(round(mid_c)) + dc
            if 0 <= r and r + sz <= h and 0 <= c and c + sz <= w:
                # Prefer a square made entirely of 1s.
                if one_count(r, c) == sz * sz:
                    best = (r, c)
                    break
        if best is not None:
            break

    # If no all-1 square near midpoint, try using the 2 square if it is near midpoint.
    if best is None:
        s = midpoint_score(r2_min, c2_min)
        if s <= 2 * sz * sz:  # reasonably close to midpoint
            best = (r2_min, c2_min)

    # If still none, search all valid square positions.
    # Score: first midpoint distance, then more 1s, then smaller row/col.
    if best is None:
        candidates = []
        for r in range(h - sz + 1):
            for c in range(w - sz + 1):
                # Do not select the original 0 square.
                if r == r0_min and c == c0_min:
                    continue
                # Do not select a square that is completely background.
                if one_count(r, c) <= 0:
                    continue
                candidates.append((midpoint_score(r, c), -one_count(r, c), r, c))
        if candidates:
            candidates.sort()
            best = (candidates[0][2], candidates[0][3])

    # Final fallback: use the 2 square.
    if best is None:
        best = (r2_min, c2_min)

    br, bc = best

    # Fill the original 0 square with 1.
    grid[r0_min:r0_min + sz, c0_min:c0_min + sz] = 1

    # Place the new 0 square at the target position.
    grid[br:br + sz, bc:bc + sz] = 0

    return grid