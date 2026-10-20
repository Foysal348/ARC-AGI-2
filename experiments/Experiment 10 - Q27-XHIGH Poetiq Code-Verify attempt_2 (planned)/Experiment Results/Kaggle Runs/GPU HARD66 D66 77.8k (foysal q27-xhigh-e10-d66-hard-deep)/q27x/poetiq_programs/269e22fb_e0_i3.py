import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete the 20x20 binary grid from a rectangular crop.

    The examples are best explained by the following rule:

    * The final answer is always a 20x20 grid.
    * The input is a rectangular crop that appears unchanged in the output.
    * The crop may start at any position in the 20x20 canvas.
    * The whole 20x20 pattern is a two-color wallpaper-like pattern whose
      rows are built from the input crop by repeating its visible row
      structure.  For rows that extend beyond the input, the output row is
      obtained by continuing the input row with the smallest repeating
      horizontal segment that is consistent with the given example crop.
      If no smaller segment is forced by the input, the last available row
      pattern is used to fill the missing width.
    * Rows above/below the input crop are completed by mirroring the input's
      row sequence with a period of (2 * input_height - 2), which matches the
      examples where the vertical continuation is symmetric.

    This is intentionally conservative:
      1. place the input into a 20x20 canvas;
      2. for every row, infer a 20-wide row by using the known input row when
         possible;
      3. when a row is missing, use vertical reflection of the input rows;
      4. when a row is too short, extend it by repeating its longest suffix
         that also occurs as a prefix of a nearby known row, falling back to
         tiling the row itself.

    The function is deterministic and total.  It is designed to preserve every
    given input crop exactly and to complete the remaining 20x20 canvas in the
    same style as the supplied examples.
    """
    grid = np.asarray(grid)

    if grid.size == 0:
        return np.zeros((20, 20), dtype=int)

    H, W = grid.shape
    N = 20

    # Determine background color.  In ARC grids, 0 is usually background if present.
    colors = np.unique(grid)
    if len(colors) == 0:
        bg = 0
    elif 0 in colors:
        bg = 0
    else:
        bg = int(colors[0])

    # We do not know where the input crop is inside the 20x20 output.
    # The examples include top-left, bottom-right, and shifted placements.
    # Try all placements that fit.  For each placement, build a candidate
    # 20x20 grid using the completion heuristics above and keep the one that
    # is most locally consistent (least forced conflict at already-known
    # neighboring crop borders).  In practice, for valid ARC inputs this
    # quickly finds a stable placement.

    def extend_row(row: np.ndarray, target_width: int) -> np.ndarray:
        """
        Extend a 1D row to target_width.

        Strategy:
        - If already long enough, return first target_width values.
        - Try to find the smallest period p that is consistent with all
          overlapping pairs in the row.
        - If a consistent period is found, use it.
        - Otherwise, tile the whole row.
        """
        row = np.asarray(row).flatten()
        L = len(row)
        if L >= target_width:
            return row[:target_width].copy()

        # Try small periods.  Prefer periods that align with the row's visible
        # repeats.  A period p is valid if row[i] == row[i+p] for all i < L-p.
        for p in range(1, L + 1):
            ok = True
            for i in range(L - p):
                if row[i] != row[i + p]:
                    ok = False
                    break
            if ok:
                out = np.empty(target_width, dtype=row.dtype)
                for j in range(target_width):
                    out[j] = row[j % p]
                return out

        # Fallback: tile the row.
        out = np.empty(target_width, dtype=row.dtype)
        for j in range(target_width):
            out[j] = row[j % L]
        return out

    def reflect_index(i: int, length: int) -> int:
        """Mirror index i into [0, length-1] with period 2*length-2."""
        if length == 1:
            return 0
        period = 2 * length - 2
        t = i % period
        if t < length:
            return t
        return period - t

    def candidate_for_placement(r0: int, c0: int) -> np.ndarray:
        """
        Build a full 20x20 candidate assuming the input starts at (r0, c0).
        """
        out = np.full((N, N), bg, dtype=grid.dtype)
        out[r0:r0 + H, c0:c0 + W] = grid

        # 1) Fill rows that are fully or partially inside the input vertically.
        #    For rows r0..r0+H-1, we know W columns.  Extend each row to 20.
        for r in range(r0, r0 + H):
            known = grid[r - r0, :]
            row20 = extend_row(known, N)

            # Preserve the known segment exactly.
            for c in range(c0, c0 + W):
                row20[c] = known[c - c0]

            # Fill unknown cells in this row.
            for c in range(N):
                if not (c0 <= c < c0 + W):
                    out[r, c] = row20[c]

        # 2) Fill rows above the input by vertical reflection of input rows.
        for r in range(0, r0):
            src_r = r0 - 1 - (r0 - 1 - r) if False else None  # placeholder
            # Mirror around the top edge of the input crop.
            # If input top is r0, row r0-1 mirrors row r0, r0-2 mirrors r0+1, etc.
            src = r0 + (r0 - 1 - r)
            if src >= r0 + H:
                # Outside the input; use reflected index within input height.
                src = r0 + reflect_index((r0 - 1 - r), H)
            row_known = grid[src - r0, :]
            row20 = extend_row(row_known, N)
            for c in range(c0, c0 + W):
                row20[c] = row_known[c - c0]
            out[r, :] = row20

        # 3) Fill rows below the input by vertical reflection.
        bottom = r0 + H
        for r in range(bottom, N):
            # Mirror around the bottom edge of the input crop.
            d = r - bottom
            src = bottom - 1 - d
            if src < r0:
                src = r0 + reflect_index(bottom - 1 - d, H)
            row_known = grid[src - r0, :]
            row20 = extend_row(row_known, N)
            for c in range(c0, c0 + W):
                row20[c] = row_known[c - c0]
            out[r, :] = row20

        # 4) Re-assert the input crop exactly (guards against overwrites).
        out[r0:r0 + H, c0:c0 + W] = grid

        return out

    def consistency_score(out: np.ndarray, r0: int, c0: int) -> float:
        """
        Prefer placements where the crop boundaries do not create abrupt
        isolated single-cell mismatches with their immediate outside neighbors.
        This is a weak prior, but it helps choose the intended offset when
        several placements are geometrically possible.
        """
        score = 0.0
        h, w = H, W

        # Penalize isolated boundary cells whose 4-neighbor outside the crop
        # is mostly the opposite color.  A correct crop usually continues
        # smoothly into the completed canvas.
        def local_penalty(r: int, c: int, val: int) -> float:
            neigh = []
            if r > 0:
                neigh.append(out[r - 1, c])
            if r + 1 < N:
                neigh.append(out[r + 1, c])
            if c > 0:
                neigh.append(out[r, c - 1])
            if c + 1 < N:
                neigh.append(out[r, c + 1])
            if not neigh:
                return 0.0
            same = sum(1 for x in neigh if x == val)
            return (len(neigh) - same) / len(neigh)

        # Top border of crop.
        if r0 > 0:
            for c in range(c0, c0 + w):
                score += local_penalty(r0, c, out[r0, c])
        # Bottom border.
        if r0 + h < N:
            for c in range(c0, c0 + w):
                score += local_penalty(r0 + h - 1, c, out[r0 + h - 1, c])
        # Left border.
        if c0 > 0:
            for r in range(r0, r0 + h):
                score += local_penalty(r, c0, out[r, c0])
        # Right border.
        if c0 + w < N:
            for r in range(r0, r0 + h):
                score += local_penalty(r, c0 + w - 1, out[r, c0 + w - 1])

        # Slight preference for placements closer to a corner, because the
        # provided examples often use corner or near-corner crops.
        corner_bias = (min(r0, N - r0 - h) + min(c0, N - c0 - w)) * 0.01
        return score + corner_bias

    best_out = None
    best_score = float("inf")

    # Try all possible placements.  For valid task inputs, one placement will
    # be clearly more consistent than the rest.
    for r0 in range(N - H + 1):
        for c0 in range(N - W + 1):
            cand = candidate_for_placement(r0, c0)
            s = consistency_score(cand, r0, c0)
            if s < best_score:
                best_score = s
                best_out = cand

    if best_out is None:
        # Very defensive fallback.
        out = np.full((N, N), bg, dtype=grid.dtype)
        ph = min(H, N)
        pw = min(W, N)
        out[:ph, :pw] = grid[:ph, :pw]
        return out

    return best_out