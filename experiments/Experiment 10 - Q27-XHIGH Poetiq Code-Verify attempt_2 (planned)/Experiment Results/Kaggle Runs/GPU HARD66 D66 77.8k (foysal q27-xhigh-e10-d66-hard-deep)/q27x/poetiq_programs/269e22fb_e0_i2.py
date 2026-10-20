import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 20x20 binary pattern from the given rectangular crop.

    The rule used by the examples is:
      1. The output is always a 20x20 grid.
      2. The input is a crop of the output.
      3. The crop may be located at any rectangle whose sides align with the
         20x20 grid.
      4. The missing parts are completed by assuming the whole 20x20 grid is a
         2D periodic extension of the input crop.
      5. If the input dimensions are odd/even in a way that the 20x20 canvas is
         smaller than two full periods, the periodic wrap-around is used.

    This implementation searches for the placement of the input inside a 20x20
    toroidal periodic extension.  For every possible top-left offset it builds
    the periodic candidate and checks whether the input matches exactly.
    The first matching candidate is returned.  If no exact match is found (for
    robustness), it returns the periodic extension from the top-left corner.
    """
    grid = np.asarray(grid)

    if grid.size == 0:
        return np.zeros((20, 20), dtype=int)

    h, w = grid.shape
    target = 20

    # The inputs in this task are two-color grids.  Keep the dtype/value as-is.
    # Build the periodic 20x20 grid from the input.
    def periodic_from_offset(r0: int, c0: int) -> np.ndarray:
        """
        Return a 20x20 grid where cell (r, c) is taken from
        grid[(r - r0) % h, (c - c0) % w].
        """
        out = np.empty((target, target), dtype=grid.dtype)
        for r in range(target):
            src_r = (r - r0) % h
            for c in range(target):
                src_c = (c - c0) % w
                out[r, c] = grid[src_r, src_c]
        return out

    # Try all possible placements of the input inside the 20x20 canvas.
    # The crop can start at any row 0..20-h and column 0..20-w.
    best_out = None
    best_score = -1

    for r0 in range(target - h + 1):
        for c0 in range(target - w + 1):
            cand = periodic_from_offset(r0, c0)

            # The input must appear unchanged at (r0, c0).
            if not np.array_equal(cand[r0:r0 + h, c0:c0 + w], grid):
                continue

            # If multiple placements are possible, prefer the one that makes
            # the whole 20x20 grid most consistent with a simple period.
            # In practice, the examples have a unique exact placement.
            # We use a small score to prefer smaller offsets (top-left first).
            score = - (r0 * 20 + c0)
            if score > best_score:
                best_score = score
                best_out = cand

            # Early exit for the first valid top-left-most placement.
            if r0 == 0 and c0 == 0:
                return cand

    if best_out is not None:
        return best_out

    # Fallback: if no exact placement is found, return the top-left periodic
    # extension.  This is not expected for valid task inputs, but keeps the
    # function total and deterministic.
    return periodic_from_offset(0, 0)